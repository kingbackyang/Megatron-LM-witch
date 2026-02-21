# WITCH on Megatron-LM: Deep Dive (Architecture, Evolution, Problem Closure)

> This document focuses on `witch_model/` (the custom training/export stack), and does not replace the official root `README.md`.

## 1. What this codebase does

This implementation extends Megatron-LM to train and export GPT-like models with **WitchAttention + GQA + MTP**, including:

- Single-node and multi-node training scripts
- Real dataset (`.bin/.idx`) discovery and build pipeline
- Custom attention module (Query Gate + Token Gate)
- MTP (Multi-Token Prediction) training path
- Hugging Face export (`safetensors`)

Primary entry points:

- Base training: `witch_model/pretrain_witch_real.py`
- Custom attention: `witch_model/witch_layers.py`
- MTP training: `witch_model/mtp_witch/pretrain_witch_real_mtp.py`
- 1B MTP training: `witch_model/mtp_witch_1b/pretrain_witch_1b_mtp.py`
- WeCar MTP training: `witch_model/wecar_mtp/pretrain_wecar_mtp.py`
- HF export: `witch_model/hf_export/convert_to_hf.py`

---

## 2. Directory layout (by responsibility)

```text
witch_model/
├── pretrain_witch_real.py                 # Mainline: real-data training (Witch + GQA)
├── witch_layers.py                        # Mainline: WitchAttention implementation
├── batch_preprocess.py                    # jsonl -> bin/idx batch preprocessing
├── training_real.sh                       # Single-node real-data run (early params)
├── training_real_gqa.sh                   # Single-node GQA run (recommended)
├── training_prefix_caching.sh             # Single-process index warmup/build
├── training_real_gqa_multinode*.sh        # Multi-node templates (2~8 node variants)
├── training_multinode_16_nnode*.sh        # 16-GPU multi-node templates
├── mtp_witch/
│   ├── pretrain_witch_real_mtp.py         # Witch + GQA + MTP
│   ├── training_real_gqa_mtp_8gpu.sh      # 8-GPU single-node MTP
│   ├── smoke_test_mtp.py                  # MTP smoke validation
│   └── smoke_test_mtp.sh
├── mtp_witch_1b/
│   ├── pretrain_witch_1b_mtp.py           # 1B path
│   ├── training_witch_1b_mtp_8gpu.sh      # 8-GPU single-node
│   ├── training_witch_1b_mtp_multinode*.sh
│   └── training_witch_1b_mtp_multinode_pdsh.sh
├── wecar_mtp/
│   ├── pretrain_wecar_mtp.py              # WeCar 1.7B + MTP + TE compatibility
│   ├── training_wecar_mtp_8gpu.sh
│   └── check_mmap.py                      # mmap compatibility checker
├── hf_export/
│   ├── convert_to_hf.py                   # Megatron ckpt -> HF safetensors
│   ├── configuration_witch.py
│   └── modeling_witch.py
└── qwen0_6B_customv2/                     # tokenizer/model assets (includes upstream README)
```

---

## 3. Mainline workflow (recommended order)

### Step A: Data preprocessing

Convert jsonl files into Megatron `.bin/.idx`:

```bash
python witch_model/batch_preprocess.py \
  --input-dir /path/to/pretrain_data \
  --tokenizer-model /path/to/Megatron-LM/witch_model/qwen0_6B_customv2 \
  --threads 8 \
  --workers 32
```

Implementation highlights:

- Recursive jsonl discovery: `batch_preprocess.py:53`
- Output root rewrite (`pretrain_data -> pretrain_data_megatron_bin`): `batch_preprocess.py:62`
- Parallel per-file processing: `batch_preprocess.py:127`

### Step B: Optional index warmup (to avoid first-run distributed timeout)

```bash
bash witch_model/training_prefix_caching.sh
```

This script forces `nproc_per_node=1` and `TP=1` to build dataset indexes in a stable way.

### Step C: Single-node training (recommended entry)

```bash
bash witch_model/training_real_gqa.sh
```

Python entry: `pretrain_witch_real.py`

### Step D: Multi-node training

Duplicate and edit `MASTER_ADDR / MASTER_PORT / NODE_RANK` per node:

- `training_real_gqa_multinode.sh`
- `training_real_gqa_multinode2.sh` ... `training_real_gqa_multinode8.sh`

### Step E: MTP training

```bash
bash witch_model/mtp_witch/training_real_gqa_mtp_8gpu.sh
```

For quick functionality validation first:

```bash
bash witch_model/mtp_witch/smoke_test_mtp.sh
```

### Step F: Export to Hugging Face

```bash
python witch_model/hf_export/convert_to_hf.py \
  --megatron-ckpt /path/to/merged/mp_rank_00_model_states.pt \
  --output-dir /path/to/hf_out \
  --vocab-size 151936 \
  --hidden-size 1024 \
  --num-layers 28 \
  --num-heads 16 \
  --num-kv-heads 8 \
  --kv-channels 128 \
  --ffn-hidden-size 3072 \
  --max-position-embeddings 40960 \
  --rotary-base 1000000 \
  --rotary-pct 1.0
```

---

## 4. Key problems solved and how

Below are the main engineering issues actually addressed in code.

### Problem 1: Inconsistent real-data path formats caused startup failures

Symptom: `--data-path` can be directory/prefix/`.bin/.idx` mixed input, which is fragile in distributed runs.

Solution:

- Unified prefix collection and dedup: `pretrain_witch_real.py:56`
- Per-prefix validation: existence, non-empty, token count: `pretrain_witch_real.py:60`
- Directory scan supports multiple `.bin/.idx` pairs: `pretrain_witch_real.py:84`

Outcome: bad inputs are filtered before training starts.

### Problem 2: Cache directory contention and rank synchronization

Symptom: multiple ranks creating cache/index artifacts at once can timeout or conflict.

Solution:

- Only builder rank creates cache dir: `pretrain_witch_real.py:112`
- Barrier synchronization before build: `pretrain_witch_real.py:119`
- Explicit `is_rank_0` callback to Builder: `pretrain_witch_real.py:137`

Outcome: stable index build and rank wait behavior.

### Problem 3: Multi-GPU batch loading deadlocks

Symptom: if each TP rank pulls iterator independently, step drift can cause hangs.

Solution:

- Load data on TP rank 0 only: `pretrain_witch_real.py:220`
- Then split for CP: `pretrain_witch_real.py:222`

Outcome: data flow aligns with Megatron best practice.

### Problem 4: Attention shape mismatch when `hidden_size != num_heads * head_dim`

Symptom: with `hidden_size=1024` and `16*128=2048`, conventional projection assumptions fail.

Solution:

- Explicit Q projection size `q_size = num_heads * kv_channels`: `witch_layers.py:62`
- Explicit K/V size `kv_size = num_kv_heads * kv_channels`: `witch_layers.py:64`
- Output projection maps true attention width back to hidden: `witch_layers.py:102`

Outcome: supports custom head-dim-first configurations.

### Problem 5: GQA KV/Q head mismatch

Symptom: with `num_kv_heads < num_attention_heads`, attention tensors do not align.

Solution:

- Compute dynamic `num_q_per_kv`: `witch_layers.py:44`
- Expand KV heads with `repeat_kv`: `witch_layers.py:172`, `witch_layers.py:229`
- Add TP divisibility assertions: `witch_layers.py:50`

Outcome: GQA works reliably under TP partitioning.

### Problem 6: RMSNorm mismatch with default LayerSpec

Symptom: Megatron default specs may use fused LayerNorm, conflicting with RMSNorm configs.

Solution:

- Inject `WrappedRMSNorm` for input/pre-MLP norms: `pretrain_witch_real.py:188`
- In MTP path, also override `enorm/hnorm/layer_norm`: `mtp_witch/pretrain_witch_real_mtp.py:207`

Outcome: normalization behavior is consistent in base and MTP branches.

### Problem 7: MTP output vs loss-mask shape mismatch

Symptom: with MTP enabled, model may return per-token loss; mask layout can be `[S,B]` or `[B,S]`.

Solution:

- Branch by `output_tensor.dim()`: `mtp_witch/pretrain_witch_real_mtp.py:254`
- Support both mask layouts explicitly: `mtp_witch/pretrain_witch_real_mtp.py:256`
- Raise clear runtime error for unknown shapes: `mtp_witch/pretrain_witch_real_mtp.py:261`

Outcome: robust, diagnosable MTP loss computation.

### Problem 8: API drift across Transformer Engine/local specs

Symptom: version/interface differences can trigger unexpected kwargs errors.

Solution:

- Signature-filtered dispatcher `_call_with_supported_kwargs`: `wecar_mtp/pretrain_wecar_mtp.py:160`
- Runtime switch between TE and local spec: `wecar_mtp/pretrain_wecar_mtp.py:165`

Outcome: one script is portable across backend/version differences.

### Problem 9: Unclear mmap behavior on different storage systems

Symptom: `mmap=True/False` compatibility varies by environment and is hard to debug in full training.

Solution:

- Standalone checker `check_mmap.py`: `wecar_mtp/check_mmap.py:31`
- Toggle `--mmap` for A/B compatibility checks: `wecar_mtp/check_mmap.py:34`

Outcome: low-cost preflight validation before large runs.

### Problem 10: Opaque Megatron -> HF key mapping during export

Symptom: export often fails with missing/unexpected keys, hard to trace.

Solution:

- Explicit per-layer mapping for Q/K/V/O, token_gate, MLP: `hf_export/convert_to_hf.py:51`
- Report missing/unexpected key summary before load: `hf_export/convert_to_hf.py:125`
- Unified `model.safetensors` output: `hf_export/convert_to_hf.py:135`

Outcome: export process is auditable and easier to debug.

---

## 5. Development evolution (timeline)

### Stage 0: Structural bring-up with dummy data

`pretrain_witch_dummy.py` used mock data + monkey patching to validate forward/backward wiring quickly.

### Stage 1: Real data via official Builder path

`pretrain_witch_real.py` removed monkey patching and switched to formal `BlendedMegatronDatasetBuilder` flow.

### Stage 2: Witch + GQA stabilization

`witch_layers.py` finalized custom attention, GQA expansion, Token Gate fusion, TP/CP alignment.

### Stage 3: MTP training closure

`mtp_witch/` and `mtp_witch_1b/` added MTP model/loss compatibility; `smoke_test_mtp.py` added minimal functional verification.

### Stage 4: Scale-out and portability

`wecar_mtp/` added TE/local compatibility; `hf_export/` completed delivery format; multi-node script coverage expanded.

---

## 6. Recommended current entry points

Use these as mainline:

- Base training: `pretrain_witch_real.py` + `training_real_gqa.sh`
- MTP: `mtp_witch/pretrain_witch_real_mtp.py` + `mtp_witch/training_real_gqa_mtp_8gpu.sh`
- 1B MTP: `mtp_witch_1b/pretrain_witch_1b_mtp.py` + `mtp_witch_1b/training_witch_1b_mtp_8gpu.sh`
- WeCar: `wecar_mtp/pretrain_wecar_mtp.py` + `wecar_mtp/training_wecar_mtp_8gpu.sh`
- Export: `hf_export/convert_to_hf.py`

Not recommended as primary entry now:

- `pretrain_witch.py`
- `pretrain_witch_dummy.py`
- `pretrain_witch_dummy_bak.py`

Reason: these scripts still import `from witch_layer import ...`, while the active file is `witch_layers.py`.

---

## 7. Current state and caveats

- `start_training.sh` and `training.sh` are currently empty files (0 lines).
- Many scripts contain hardcoded paths (`/data2/...`, `/apdcephfs_...`); adjust before migration.
- Large assets (`qwen0_6B_customv2/model.safetensors`, `qwen_processed_text_document.bin`) should be managed separately from source code.
- Multi-node scripts are maintained as per-node variants (`multinode2...8`); major differences are mostly `NODE_RANK/NNODES`.

---

## 8. Minimal pre-run checklist

1. Run syntax checks (`python -m py_compile`).
2. Run `check_mmap.py` on target data paths.
3. Ensure `training_prefix_caching.sh` reaches `Datasets built successfully` once.
4. Validate 1-10 single-node steps before long multi-node training.
5. After export, run a minimal `transformers` generation sanity test.

---

## 9. One-line summary

`witch_model/` now forms a complete engineering pipeline from preprocessing and distributed training to MTP extension and HF export, with code-level closure on the highest-risk Megatron issues (data build/sync, parallel batch flow, GQA dimensions, MTP loss compatibility, and backend/version drift).
