#!/usr/bin/env python
"""
Convert merged Megatron Witch checkpoints to HuggingFace format.

Usage:
  python convert_to_hf.py \
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
    [--rotary-base 1000000] [--rotary-pct 1.0]
"""
import argparse
import os
import torch
from safetensors.torch import save_file

from configuration_witch import WitchConfig
from modeling_witch import WitchForCausalLM


def load_megatron_state(path):
    sd = torch.load(path, map_location="cpu")
    if "model" in sd:
        sd = sd["model"]
    return sd


def map_state_dict(mg_sd, cfg: WitchConfig):
    hf_sd = {}

    def assign(hf_key, mg_key):
        if mg_key in mg_sd:
            hf_sd[hf_key] = mg_sd[mg_key]

    # Embeddings
    assign("model.embed_tokens.weight", "language_model.embedding.word_embeddings.weight")

    # Final norm
    assign("model.final_layernorm.weight", "language_model.transformer.final_layernorm.weight")

    # LM head tied; no need to copy separately (will tie later)

    # Transformer layers
    for i in range(cfg.num_hidden_layers):
        base_mg = f"language_model.encoder.layers.{i}."
        base_hf = f"model.layers.{i}."

        # LayerNorms
        assign(base_hf + "input_layernorm.weight", base_mg + "input_layernorm.weight")
        assign(base_hf + "pre_mlp_layernorm.weight", base_mg + "pre_mlp_layernorm.weight")

        # Attention projections
        assign(base_hf + "self_attention.q_proj.weight", base_mg + "self_attention.q_proj.weight")
        assign(base_hf + "self_attention.q_proj.bias", base_mg + "self_attention.q_proj.bias")
        assign(base_hf + "self_attention.k_proj.weight", base_mg + "self_attention.k_proj.weight")
        assign(base_hf + "self_attention.k_proj.bias", base_mg + "self_attention.k_proj.bias")
        assign(base_hf + "self_attention.v_proj.weight", base_mg + "self_attention.v_proj.weight")
        assign(base_hf + "self_attention.v_proj.bias", base_mg + "self_attention.v_proj.bias")
        assign(base_hf + "self_attention.o_proj.weight", base_mg + "self_attention.o_proj.weight")
        assign(base_hf + "self_attention.o_proj.bias", base_mg + "self_attention.o_proj.bias")

        # Token gate
        assign(base_hf + "self_attention.token_gate.0.weight", base_mg + "self_attention.token_gate.0.weight")
        assign(base_hf + "self_attention.token_gate.0.bias", base_mg + "self_attention.token_gate.0.bias")
        assign(base_hf + "self_attention.token_gate.2.weight", base_mg + "self_attention.token_gate.2.weight")
        assign(base_hf + "self_attention.token_gate.2.bias", base_mg + "self_attention.token_gate.2.bias")

        # MLP
        assign(base_hf + "mlp.dense_h_to_4h.weight", base_mg + "mlp.linear_fc1.weight")
        assign(base_hf + "mlp.dense_h_to_4h.bias", base_mg + "mlp.linear_fc1.bias")
        assign(base_hf + "mlp.dense_4h_to_h.weight", base_mg + "mlp.linear_fc2.weight")
        assign(base_hf + "mlp.dense_4h_to_h.bias", base_mg + "mlp.linear_fc2.bias")

    return hf_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-ckpt", required=True, help="Merged Megatron checkpoint (pt)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--kv-channels", type=int, required=True)
    parser.add_argument("--ffn-hidden-size", type=int, required=True)
    parser.add_argument("--max-position-embeddings", type=int, required=True)
    parser.add_argument("--rotary-base", type=int, default=10000)
    parser.add_argument("--rotary-pct", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = WitchConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        kv_channels=args.kv_channels,
        ffn_hidden_size=args.ffn_hidden_size,
        max_position_embeddings=args.max_position_embeddings,
        rotary_base=args.rotary_base,
        rotary_pct=args.rotary_pct,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=None,
    )

    print("Loading Megatron checkpoint...")
    mg_sd = load_megatron_state(args.megatron_ckpt)

    print("Building HF model...")
    hf_model = WitchForCausalLM(cfg)
    hf_sd = map_state_dict(mg_sd, cfg)

    missing = set(hf_model.state_dict().keys()) - set(hf_sd.keys())
    unexpected = set(hf_sd.keys()) - set(hf_model.state_dict().keys())
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (showing first 20): {list(missing)[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (showing first 20): {list(unexpected)[:20]}")

    hf_model.load_state_dict(hf_sd, strict=False)
    cfg.save_pretrained(args.output_dir)

    # Save safetensors
    tensors = {k: v.cpu() for k, v in hf_model.state_dict().items()}
    save_file(tensors, os.path.join(args.output_dir, "model.safetensors"))
    print(f"Saved HF model to {args.output_dir}")


if __name__ == "__main__":
    main()
