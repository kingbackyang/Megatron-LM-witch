# File: pretrain_witch_1b_mtp.py
import os
import sys
import torch

from megatron.training import get_args, print_rank_0, get_tokenizer
from megatron.training import pretrain
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig

# Allow importing witch_layers from the parent directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WITCH_MODEL_DIR = os.path.dirname(_THIS_DIR)
if _WITCH_MODEL_DIR not in sys.path:
    sys.path.insert(0, _WITCH_MODEL_DIR)

from witch_layers import MegatronWitchAttention  # noqa: E402


class WrappedRMSNorm(torch.nn.Module):
    def __init__(self, config, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(dtype)


def train_valid_test_datasets_provider(num_samples):
    """Build train/valid/test datasets using the MCore builder."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0("Building GPT datasets for MCore builder ...")

    def _collect_prefixes(paths):
        prefixes = []
        seq_len_required = args.seq_length + 1  # add_extra_token_to_sequence defaults to True

        def _valid_prefix(prefix):
            bin_path = prefix + ".bin"
            idx_path = prefix + ".idx"
            if not (os.path.isfile(idx_path) and os.path.isfile(bin_path)):
                print_rank_0(f"  [skip] missing bin/idx: {bin_path}, {idx_path}")
                return False
            if os.path.getsize(bin_path) == 0 or os.path.getsize(idx_path) == 0:
                print_rank_0(f"  [skip] empty bin/idx: {bin_path}, {idx_path}")
                return False
            # Quick filter: at least one sequence length.
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

    print_rank_0("Datasets built successfully via builder.")

    return datasets[0], datasets[1], datasets[2]


def model_provider(pre_process=True, post_process=True, **kwargs):
    args = get_args()
    print_rank_0("building Witch-1B GPT model with MTP ...")

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        kv_channels=args.kv_channels,
        use_cpu_initialization=args.use_cpu_initialization,
        normalization=args.normalization,
        layernorm_epsilon=args.norm_epsilon,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        params_dtype=args.params_dtype,
        fp16=args.fp16,
        bf16=args.bf16,
        apply_query_key_layer_scaling=getattr(args, "apply_query_key_layer_scaling", False),
        attention_softmax_in_fp32=getattr(args, "attention_softmax_in_fp32", True),
        bias_activation_fusion=getattr(args, "bias_activation_fusion", False),
        masked_softmax_fusion=getattr(args, "masked_softmax_fusion", False),
        mtp_num_layers=args.mtp_num_layers,
        mtp_loss_scaling_factor=args.mtp_loss_scaling_factor,
    )

    transformer_layer_spec = get_gpt_layer_local_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
    )

    if args.normalization == "RMSNorm":
        transformer_layer_spec.submodules.input_layernorm = ModuleSpec(
            module=WrappedRMSNorm
        )
        transformer_layer_spec.submodules.pre_mlp_layernorm = ModuleSpec(
            module=WrappedRMSNorm
        )

    transformer_layer_spec.submodules.self_attention = ModuleSpec(
        module=MegatronWitchAttention,
        params={"attn_mask_type": AttnMaskType.causal},
    )

    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config,
            spec=transformer_layer_spec,
            use_transformer_engine=False,
        )
        if args.normalization == "RMSNorm" and mtp_block_spec is not None:
            # MTP norms default to backend layer norm (fused). Override to RMSNorm.
            for layer_spec in mtp_block_spec.layer_specs:
                layer_spec.submodules.enorm = ModuleSpec(module=WrappedRMSNorm)
                layer_spec.submodules.hnorm = ModuleSpec(module=WrappedRMSNorm)
                layer_spec.submodules.layer_norm = ModuleSpec(module=WrappedRMSNorm)

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
        mtp_block_spec=mtp_block_spec,
    )
    return model


def get_batch(data_iterator):
    """Load batch on TP rank 0 and broadcast to other TP ranks."""
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model):
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    args = get_args()
    if args.mtp_num_layers is not None:
        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
    else:
        output_tensor = model(tokens, position_ids, attention_mask)

    def loss_func(output_tensor):
        if output_tensor.dim() == 2:
            # GPTModel returns per-token loss when labels are provided.
            if output_tensor.shape == loss_mask.shape:
                loss_mask_local = loss_mask
            elif output_tensor.shape == loss_mask.transpose(0, 1).shape:
                loss_mask_local = loss_mask.transpose(0, 1)
            else:
                raise RuntimeError(
                    f"loss shape mismatch: loss={tuple(output_tensor.shape)} "
                    f"mask={tuple(loss_mask.shape)}"
                )
            loss = torch.sum(output_tensor * loss_mask_local) / loss_mask_local.sum()
            return loss, {"lm loss": loss}

        losses = tensor_parallel.vocab_parallel_cross_entropy(
            output_tensor.float(), labels
        )
        loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / loss_mask.sum()
        return loss, {"lm loss": loss}

    return output_tensor, loss_func


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    wandb_project = os.getenv("WANDB_PROJECT", "")
    wandb_exp_name = os.getenv("WANDB_NAME", "")
    wandb_save_dir = os.getenv("WANDB_DIR", "")

    wandb_args = {}
    if wandb_project and wandb_exp_name:
        os.environ.setdefault("WANDB_MODE", "offline")
        wandb_args = {
            "wandb_project": wandb_project,
            "wandb_exp_name": wandb_exp_name,
            "wandb_save_dir": wandb_save_dir,
        }

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults=wandb_args,
    )
