# File: pretrain_wecar_mtp.py
import os
import inspect
import torch
from functools import partial

from megatron.training import get_args, print_rank_0, get_tokenizer
from megatron.training import pretrain
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)

_DEBUG_STATE = {"step": 0}


def train_valid_test_datasets_provider(num_samples):
    """Build train/valid/test datasets using the MCore builder."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0("Building WeCar GPT datasets (MCore builder) ...")

    def _collect_prefixes(paths):
        prefixes = []
        seq_len_required = args.seq_length + 1

        def _valid_prefix(prefix):
            bin_path = prefix + ".bin"
            idx_path = prefix + ".idx"
            if not (os.path.isfile(idx_path) and os.path.isfile(bin_path)):
                print_rank_0(f"  [skip] missing bin/idx: {bin_path}, {idx_path}")
                return False
            if os.path.getsize(bin_path) == 0 or os.path.getsize(idx_path) == 0:
                print_rank_0(f"  [skip] empty bin/idx: {bin_path}, {idx_path}")
                return False
            from megatron.core.datasets.indexed_dataset import IndexedDataset
            try:
                ds = IndexedDataset(prefix, multimodal=False, mmap=False)
                num_tokens = int(ds.sequence_lengths.sum())
                if num_tokens < seq_len_required:
                    print_rank_0(
                        f"  [skip] tokens {num_tokens} < seq_len {seq_len_required}: {prefix}"
                    )
                    return False
            except Exception as exc:
                print_rank_0(f"  [skip] failed to open {prefix}: {exc}")
                return False
            return True

        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for fname in files:
                        if fname.endswith(".bin"):
                            prefix = os.path.join(root, fname[:-4])
                            if _valid_prefix(prefix):
                                prefixes.append(prefix)
            else:
                prefix = path
                if prefix.endswith(".bin") or prefix.endswith(".idx"):
                    prefix = prefix[:-4]
                if _valid_prefix(prefix):
                    prefixes.append(prefix)

        prefixes = sorted(set(prefixes))
        if not prefixes:
            raise RuntimeError(f"No valid .bin/.idx prefixes found from data-paths: {paths}")
        return prefixes

    data_prefixes = _collect_prefixes(args.data_path)
    print_rank_0(f"Data Prefixes ({len(data_prefixes)}):")
    for prefix in data_prefixes:
        print_rank_0(f"  - {prefix}")

    cache_dir = args.data_cache_path
    if cache_dir:
        is_builder_rank = (
            parallel_state.is_pipeline_first_stage()
            and parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_data_parallel_rank() == 0
        )
        if is_builder_rank:
            os.makedirs(cache_dir, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    dataset_config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=(data_prefixes, None),
        split=args.split,
        path_to_cache=args.data_cache_path,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
    )

    def is_rank_0():
        return (
            parallel_state.is_pipeline_first_stage()
            and parallel_state.get_tensor_model_parallel_rank() == 0
        )

    builder = BlendedMegatronDatasetBuilder(
        GPTDataset,
        num_samples,
        is_rank_0,
        dataset_config,
    )
    datasets = builder.build()

    print_rank_0("Datasets built successfully.")
    return datasets[0], datasets[1], datasets[2]


def model_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):
    args = get_args()
    print_rank_0("building WeCar GPT model with MTP ...")

    config = core_transformer_config_from_args(args)
    print_rank_0(
        "Model config: layers={}, hidden={}, heads={}, kv_channels={}, num_query_groups={}, tp={}, pp={}, transformer_impl={}, mtp_layers={}, norm={}".format(
            config.num_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.kv_channels,
            config.num_query_groups,
            config.tensor_model_parallel_size,
            config.pipeline_model_parallel_size,
            args.transformer_impl,
            config.mtp_num_layers,
            config.normalization,
        )
    )

    use_te = args.transformer_impl == "transformer_engine"
    use_kitchen = getattr(config, "use_kitchen", False)
    use_kitchen_attention = getattr(config, "use_kitchen_attention", False)
    kitchen_attention_backend = getattr(config, "kitchen_attention_backend", "sdpa")

    def _call_with_supported_kwargs(func, **call_kwargs):
        sig = inspect.signature(func)
        filtered = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
        return func(**filtered)

    if use_te:
        transformer_layer_spec = _call_with_supported_kwargs(
            get_gpt_layer_with_transformer_engine_spec,
            num_experts=args.num_experts,
            moe_grouped_gemm=args.moe_grouped_gemm,
            qk_layernorm=args.qk_layernorm,
            multi_latent_attention=args.multi_latent_attention,
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=use_kitchen,
            use_kitchen_attention=use_kitchen_attention,
            kitchen_attention_backend=kitchen_attention_backend,
        )
    else:
        transformer_layer_spec = _call_with_supported_kwargs(
            get_gpt_layer_local_spec,
            num_experts=args.num_experts,
            moe_grouped_gemm=args.moe_grouped_gemm,
            qk_layernorm=args.qk_layernorm,
            multi_latent_attention=args.multi_latent_attention,
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=use_kitchen,
            use_kitchen_attention=use_kitchen_attention,
            kitchen_attention_backend=kitchen_attention_backend,
        )

    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config,
            spec=transformer_layer_spec,
            use_transformer_engine=use_te,
            vp_stage=vp_stage,
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=getattr(args, "use_rope_scaling", None),
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )
    return model


def get_batch(data_iterator):
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


def _debug_validate_batch(tokens, labels, loss_mask, attention_mask, position_ids):
    if tokens is None:
        print_rank_0("Debug: tokens is None (non-first/last pipeline stage).")
        return
    args = get_args()
    if tokens.dim() != 2 or labels.dim() != 2:
        raise RuntimeError(
            f"Unexpected token/label dims: tokens={tuple(tokens.shape)} labels={tuple(labels.shape)}"
        )
    if labels.shape != tokens.shape:
        raise RuntimeError(
            f"Label shape mismatch: tokens={tuple(tokens.shape)} labels={tuple(labels.shape)}"
        )
    if loss_mask.shape != labels.shape:
        raise RuntimeError(
            f"Loss mask shape mismatch: labels={tuple(labels.shape)} mask={tuple(loss_mask.shape)}"
        )
    if position_ids.shape != tokens.shape:
        raise RuntimeError(
            f"Position ids shape mismatch: tokens={tuple(tokens.shape)} pos={tuple(position_ids.shape)}"
        )
    if attention_mask is not None and attention_mask.dim() not in (2, 3, 4):
        raise RuntimeError(
            f"Unexpected attention mask dims: mask={tuple(attention_mask.shape)}"
        )
    if tokens.max().item() >= args.padded_vocab_size:
        raise RuntimeError(
            f"Token id out of range: max={tokens.max().item()} vocab={args.padded_vocab_size}"
        )
    print_rank_0(
        "Batch debug: tokens={}, labels={}, loss_mask={}, attention_mask={}, position_ids={}".format(
            tuple(tokens.shape),
            tuple(labels.shape),
            tuple(loss_mask.shape),
            None if attention_mask is None else tuple(attention_mask.shape),
            tuple(position_ids.shape),
        )
    )


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    return loss, num_tokens, {"lm loss": torch.cat([loss.detach().view(1), num_tokens.view(1)])}


def forward_step(data_iterator, model):
    _DEBUG_STATE["step"] += 1
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    if _DEBUG_STATE["step"] <= 2:
        _debug_validate_batch(tokens, labels, loss_mask, attention_mask, position_ids)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
    if _DEBUG_STATE["step"] <= 2:
        print_rank_0(
            "Output debug: output_tensor shape={}, dtype={}".format(
                tuple(output_tensor.shape), output_tensor.dtype
            )
        )
    return output_tensor, partial(loss_func, loss_mask)


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={
            "tokenizer_type": "HuggingFaceTokenizer",
            "tokenizer_model": "witch_model/qwen0_6B_customv2",
        },
    )
