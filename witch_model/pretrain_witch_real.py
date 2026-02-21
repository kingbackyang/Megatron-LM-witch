# 文件名: pretrain_witch_real.py
import os
import torch
from functools import partial

from megatron.training import get_args, print_rank_0, get_tokenizer
from megatron.training import pretrain
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
# 🟢 引入 MCore v0.14.0 的数据集模块
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.utils import Split

# 导入你的模型（GQA 版本在 witch_layers.py）
from witch_layers import MegatronWitchAttention

# ==========================================
# 0. 辅助类 (RMSNorm)
# ==========================================
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

# ==========================================
# 1. 真实数据集 Provider (v0.14.0)
# ==========================================
def train_valid_test_datasets_provider(num_samples):
    """
    使用 MCore v0.14.0 的 Builder 模式自动构建 Train/Valid/Test 数据集
    """
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0(f"> Building Real GPT Datasets for MCore v0.14.0 using Builder ...")

    # 支持目录下多组 .bin/.idx：遍历收集前缀列表
    def _collect_prefixes(paths):
        prefixes = []
        seq_len_required = args.seq_length + 1  # add_extra_token_to_sequence 默认为 True

        def _valid_prefix(prefix):
            bin_path = prefix + ".bin"
            idx_path = prefix + ".idx"
            if not (os.path.isfile(idx_path) and os.path.isfile(bin_path)):
                print_rank_0(f"  [skip] missing bin/idx: {bin_path}, {idx_path}")
                return False
            if os.path.getsize(bin_path) == 0 or os.path.getsize(idx_path) == 0:
                print_rank_0(f"  [skip] empty bin/idx: {bin_path}, {idx_path}")
                return False
            # 粗过滤：至少满足一个序列长度
            from megatron.core.datasets.indexed_dataset import IndexedDataset
            try:
                ds = IndexedDataset(prefix, multimodal=False, mmap=False)
                num_tokens = int(ds.sequence_lengths.sum())
                if num_tokens < seq_len_required:
                    print_rank_0(
                        f"  [skip] tokens {num_tokens} < seq_len {seq_len_required}: {prefix}"
                    )
                    return False
            except Exception as e:
                print_rank_0(f"  [skip] failed to open {prefix}: {e}")
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
        # 去重并排序，避免重复
        prefixes = sorted(set(prefixes))
        if not prefixes:
            raise RuntimeError(f"No valid .bin/.idx prefixes found from data-paths: {paths}")
        return prefixes

    data_prefixes = _collect_prefixes(args.data_path)
    print_rank_0(f"> Data Prefixes ({len(data_prefixes)}):")
    for p in data_prefixes:
        print_rank_0(f"  - {p}")

    # 确保缓存目录可写
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
        eod_mask_loss=args.eod_mask_loss
    )

    # 🟢 关键修复：定义 is_built_on_rank 函数
    # 只有全局 Rank 0 返回 True，其他返回 False
    def is_rank_0():
        return (parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0)

    builder = BlendedMegatronDatasetBuilder(
        GPTDataset,
        num_samples,
        is_rank_0, # <--- 传入这个函数
        dataset_config
    )

    # build() 内部会自动处理同步：Rank 0 加载，Rank 1-7 等待
    datasets = builder.build()

    print_rank_0("> Datasets built successfully via Builder.")
    
    return datasets[0], datasets[1], datasets[2]


# ==========================================
# 2. 模型构建 (Model Provider)
# ==========================================
def model_provider(pre_process=True, post_process=True, **kwargs):
    args = get_args()
    print_rank_0("building Witch GPT model ...")

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
        apply_query_key_layer_scaling=getattr(args, 'apply_query_key_layer_scaling', False),
        attention_softmax_in_fp32=getattr(args, 'attention_softmax_in_fp32', True),
        bias_activation_fusion=getattr(args, 'bias_activation_fusion', False),
        masked_softmax_fusion=getattr(args, 'masked_softmax_fusion', False),
    )

    transformer_layer_spec = get_gpt_layer_local_spec(
        num_experts=args.num_experts, 
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
    )

    if args.normalization == 'RMSNorm':
        transformer_layer_spec.submodules.input_layernorm = ModuleSpec(module=WrappedRMSNorm)
        transformer_layer_spec.submodules.pre_mlp_layernorm = ModuleSpec(module=WrappedRMSNorm)

    transformer_layer_spec.submodules.self_attention = ModuleSpec(
        module=MegatronWitchAttention,
        params={'attn_mask_type': AttnMaskType.causal},
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
        rotary_percent=args.rotary_percent
    )
    return model

# ==========================================
# 3. Forward Step
# ==========================================
def get_batch(data_iterator):
    """
    只在 TP rank 0 拉取数据，再广播到其余 TP rank，避免多卡卡死。
    """
    # 参考官方 GPT 实现，先在 tp_rank0 取数据并广播
    batch = get_batch_on_this_tp_rank(data_iterator)
    # 如果使用了 CP，需要再切分
    batch = get_batch_on_this_cp_rank(batch)
    tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model):
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    output_tensor = model(tokens, position_ids, attention_mask)

    def loss_func(output_tensor):
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            output_tensor.float(), labels
        )
        loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / loss_mask.sum()
        return loss, {'lm loss': loss}

    return output_tensor, loss_func

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    # 与官方脚本一致，标记分布式数据集构建
    train_valid_test_datasets_provider.is_distributed = True

    # WandB 离线记录：运行前设置 WANDB_PROJECT / WANDB_NAME，可选 WANDB_DIR
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

    # 无需 Monkey Patch，正规启动
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        # 这里不需要覆盖 tokenizer_type，由脚本控制；按需透传 wandb
        args_defaults=wandb_args,
    )
