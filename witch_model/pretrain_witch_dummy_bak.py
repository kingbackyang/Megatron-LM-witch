# 文件名: pretrain_witch_dummy.py
import torch
from torch.utils.data import Dataset
from functools import partial

from megatron.training import get_args, print_rank_0
from megatron.training import pretrain
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core.transformer.transformer_config import TransformerConfig
# 导入你的模型
from witch_layer import MegatronWitchAttention

# ==========================================
# 手动定义 RMSNorm (适配 Megatron 接口)
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
# 1. 定义虚拟数据集 (Dummy Data)
# ==========================================
class MockGPTDataset(Dataset):
    """一个无限生成随机整数的数据集，用来测试模型能不能跑通"""
    def __init__(self, length, seq_len, vocab_size):
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 返回 seq_len + 1 个 token (input + label)
        # Megatron 的 dataset 约定返回字典 {'text': np.array or torch.tensor}
        data = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return {'text': data}

def fake_train_valid_test_datasets_provider(train_val_test_num_samples):
    """替代真正的数据提供者，返回 3 个虚拟数据集"""
    args = get_args()
    print_rank_0(f"> Building Dummy Datasets for Debugging...")
    
    # 造一个足够大的虚拟数据集
    dummy_ds = MockGPTDataset(
        length=100000, 
        seq_len=args.seq_length, 
        vocab_size=args.padded_vocab_size
    )
    # Train, Valid, Test 都用同一个虚拟集
    return [dummy_ds], [dummy_ds], [dummy_ds]

# ==========================================
# 2. 定义模型构建器 (Model Provider)
# ==========================================
def model_provider(pre_process=True, post_process=True, **kwargs):
    args = get_args()
    print_rank_0("building Witch GPT model ...")
    
    # --- 1. 补全 args 的初始化函数 (保持之前的修复) ---
    if not hasattr(args, 'init_method'):
        import torch
        std = getattr(args, 'init_method_std', 0.02)
        def _init_method(tensor):
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)
        args.init_method = _init_method

    # --- 2. 【核心修复】创建 TransformerConfig 对象 ---
    # Megatron-Core 要求必须用这个类，而不是 Namespace
    # --- 2. 【核心修复】创建 TransformerConfig 对象 ---
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        kv_channels=args.kv_channels, 
        
        # 初始化与正则化
        init_method=args.init_method,
        output_layer_init_method=args.init_method,
        use_cpu_initialization=args.use_cpu_initialization,
        normalization=args.normalization,
        
        # 🟢 修正：使用 layernorm_epsilon
        layernorm_epsilon=args.norm_epsilon, 
        
        # 并行参数
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        
        # 数据类型
        params_dtype=args.params_dtype,

        fp16=args.fp16,
        bf16=args.bf16,
        
        # 其他开关
        apply_query_key_layer_scaling=getattr(args, 'apply_query_key_layer_scaling', False),
        attention_softmax_in_fp32=getattr(args, 'attention_softmax_in_fp32', True),
        bias_activation_fusion=getattr(args, 'bias_activation_fusion', False),
        masked_softmax_fusion=getattr(args, 'masked_softmax_fusion', False),
        # persistence_bias_in_fp32=getattr(args, 'accumulate_allreduce_grads_in_fp32', False),
    )

    # --- 3. 构建 Spec ---
    transformer_layer_spec = get_gpt_layer_local_spec(
        num_experts=args.num_experts, 
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
    )

    # 🟢【新增修复】替换 Norm 层为 RMSNorm
    # 默认 Spec 使用 FusedLayerNorm，它不支持 RMSNorm 配置，必须替换
    if args.normalization == 'RMSNorm':
        transformer_layer_spec.submodules.input_layernorm = ModuleSpec(module=WrappedRMSNorm)
        transformer_layer_spec.submodules.pre_mlp_layernorm = ModuleSpec(module=WrappedRMSNorm)

    # 注入 Witch Attention
    transformer_layer_spec.submodules.self_attention = ModuleSpec(
        module=MegatronWitchAttention,
        params={'attn_mask_type': AttnMaskType.causal},
    )

    # --- 4. 实例化模型 (传入 config 而不是 args) ---
    model = GPTModel(
        config=config,  # <--- 这里传入刚创建的 TransformerConfig 对象
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

def model_provider_v2(pre_process=True, post_process=True, **kwargs):
    args = get_args()
    print_rank_0("building Witch GPT model ...")

    # 1. 如果 args 只有数值 (init_method_std) 而没有函数 (init_method)，我们自己造一个
    if not hasattr(args, 'init_method'):
        # 获取标准差，默认 0.02
        std = getattr(args, 'init_method_std', 0.02)
        # 定义一个高斯初始化函数
        def _init_method(tensor):
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)
        args.init_method = _init_method 

    # ----------------------------------------------------
    # 🛠️ 修复：手动修补 args，补充缺失的配置属性
    # ----------------------------------------------------
    if not hasattr(args, 'embedding_init_method'):
        # 如果 args 里没有定义 embedding 初始化方法，就用默认的 init_method
        args.embedding_init_method = args.init_method

    # 某些新版本可能还需要 output_layer_init_method，保险起见也补上
    if not hasattr(args, 'output_layer_init_method'):
        args.output_layer_init_method = args.init_method
    # ----------------------------------------------------

    # 获取默认 Spec
    transformer_layer_spec = get_gpt_layer_local_spec(
        num_experts=args.num_experts, 
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
    )

    # 【注入】将 Self Attention 替换为 Witch Attention
    transformer_layer_spec.submodules.self_attention = ModuleSpec(
        module=MegatronWitchAttention,
        params={'attn_mask_type': AttnMaskType.causal},
    )

    model = GPTModel(
        config=args,
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
# 3. 定义 Forward Step (计算 Loss)
# ==========================================
def get_batch(data_iterator):
    """
    从迭代器获取数据，广播给组内所有 GPU，并生成 mask。
    """
    args = get_args()
    
    # 1. 定义需要广播的数据键和类型
    # 我们的 MockDataset 返回的是 {'text': tensor(...)}
    keys = ['text']
    datatype = torch.int64

    # 2. 从迭代器读取数据
    # 注意：在 TP 模式下，通常只有 TP 组的第一个 rank (Rank 0) 能从 iterator 读到数据，
    # 其他 rank 读到的可能是 None，或者 iterator 本身就是 None。
    data = None
    if data_iterator is not None:
        try:
            data = next(data_iterator)
        except StopIteration:
            # 如果数据读完了，这里捕获异常，data 保持为 None
            pass

    # 3. 🟢【核心修复】广播数据 (Broadcasting)
    # 这行代码会自动处理：
    # - 如果当前进程拿到数据，它会发给同组的其他进程。
    # - 如果当前进程没拿到数据（是 None），它会等待接收。
    # 结果：data_b 在所有 TP Rank 上都是有值的！
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # 4. 如果广播后依然是 None，说明整个组都没数据了（可能是跑完了）
    if data_b is None:
        return None, None, None, None, None

    # 5. 解包数据并处理形状
    # data_b['text'] 原始形状通常是 [batch, seq+1]
    tokens = data_b['text'].long().cuda()
    
    # 转置为 [seq+1, batch] (Megatron 标准格式)
    tokens = tokens.transpose(0, 1).contiguous()

    # 6. 切分 input (前 seq 个) 和 label (后 seq 个)
    # 对应: Input: "A B C D", Label: "B C D E"
    labels = tokens[1:].contiguous()
    tokens = tokens[:-1].contiguous()

    # 7. 生成 Attention Mask 和 Position IDs
    # 使用你刚才修复好的参数调用方式
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=tokenizer.eod,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        pad_token=tokenizer.pad, # 确保这里的 tokenizer.pad 是有效的 (int)
        pad_mask_loss=False
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batchv2(data_iterator):
    """从迭代器获取数据并生成 mask"""
    args = get_args()
    
    # 注意：MockDataset 已经在本地生成了 tensor，这里我们需要模拟 Megatron 的行为
    # 如果是 TP/PP，通常 data_iterator 在所有 rank 上都会 yield 数据
    # 为了简化，假设所有 rank 都能直接拿到数据
    
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        return None, None, None, None, None

    # data['text'] 形状是 [batch, seq+1]
    # 需要转为 [seq+1, batch]
    tokens = data['text'].long().cuda().transpose(0, 1).contiguous()

    # 切分 input 和 label
    labels = tokens[1:].contiguous()
    tokens = tokens[:-1].contiguous()

    # 生成 Attention Mask 和 Position IDs
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=tokenizer.eod,
        pad_token=tokenizer.pad,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        pad_mask_loss=False # pad_mask_loss，通常给 False 即可
    )

    # attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
    #     tokens,
    #     tokenizer.eod if hasattr(tokenizer, 'eod') else tokenizer.vocab_size - 1,
    #     args.reset_position_ids,
    #     args.reset_attention_mask,
    #     args.eod_mask_loss
    # )

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
    # 为了让 get_batch 能访问 tokenizer 信息，我们需要 mock 一个 tokenizer
    # 或者给 arguments 塞一个假的 tokenizer
    class MockTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.eod = vocab_size - 1
            self.pad = vocab_size - 1
    
    # 稍微 hack 一下，把 tokenizer 放到全局，供 get_batch 使用
    # 实际 Megatron 中 tokenizer 是全局单例
    global tokenizer
    tokenizer = MockTokenizer(1000) # 先占位，后面 args 解析完会更新

    # 启动预训练
    pretrain(
        fake_train_valid_test_datasets_provider, # 使用虚拟数据
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'NullTokenizer'} # 使用空分词器
    )
