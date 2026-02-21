# 文件名: pretrain_witch_dummy.py
import torch
from torch.utils.data import Dataset, DataLoader
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
# 0. 辅助类定义
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

class MockGPTDataset(Dataset):
    """虚拟数据集"""
    def __init__(self, length, seq_len, vocab_size):
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 随机生成数据
        data = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return {'text': data}

class MockTokenizer:
    """虚拟分词器"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.eod = vocab_size - 1
        self.pad = vocab_size - 1

# ==========================================
# 1. Monkey Patch 工具函数
# ==========================================
def cyclic_iter(dataset, batch_size):
    """无限循环的数据迭代器"""
    # num_workers=0 防止多进程死锁，pin_memory=True 加速
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=0,
        pin_memory=True 
    )
    while True:
        for batch in loader:
            yield batch

def fake_build_train_valid_test_data_iterators(train_valid_test_dataset_provider):
    """
    🟢【核心魔法 v2.0】
    适配 Megatron v0.14.0 的 RerunDataIterator 强制检查
    """
    # 引入必要的 Wrapper 类
    from megatron.core.rerun_state_machine import RerunDataIterator
    
    args = get_args()
    print_rank_0("!!! WARNING: BYPASSING MEGATRON DATA BUILDER !!!")
    print_rank_0("!!! Forcing infinite dummy iterators (Wrapped in RerunDataIterator) !!!")
    
    # 构造假数据集
    dummy_ds = MockGPTDataset(
        length=10000, 
        seq_len=args.seq_length, 
        vocab_size=args.padded_vocab_size
    )
    
    # 获取 batch size (micro batch size)
    bs = args.micro_batch_size
    
    # 构造原始迭代器
    raw_train_iter = cyclic_iter(dummy_ds, bs)
    raw_valid_iter = cyclic_iter(dummy_ds, bs)
    raw_test_iter = cyclic_iter(dummy_ds, bs)
    
    # 🟢 关键修复：用 RerunDataIterator 包装原始迭代器
    # 这样才能通过 v0.14.0 的 _sanitize_data_iterators 检查
    train_iter = RerunDataIterator(raw_train_iter)
    valid_iter = RerunDataIterator(raw_valid_iter)
    test_iter = RerunDataIterator(raw_test_iter)
    
    # 返回列表格式
    return train_iter, [valid_iter], [test_iter]

# ==========================================
# 2. 模型构建 (Model Provider)
# ==========================================
def model_provider(pre_process=True, post_process=True, **kwargs):
    args = get_args()
    print_rank_0("building Witch GPT model ...")
    
    if not hasattr(args, 'init_method'):
        import torch
        std = getattr(args, 'init_method_std', 0.02)
        def _init_method(tensor):
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)
        args.init_method = _init_method

    # 兼容 v0.14.0 的 Config 构建
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        kv_channels=args.kv_channels, 
        init_method=args.init_method,
        output_layer_init_method=args.init_method,
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

    # 替换 Norm
    if args.normalization == 'RMSNorm':
        transformer_layer_spec.submodules.input_layernorm = ModuleSpec(module=WrappedRMSNorm)
        transformer_layer_spec.submodules.pre_mlp_layernorm = ModuleSpec(module=WrappedRMSNorm)

    # 替换 Attention 为 WitchAttention
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
    args = get_args()
    keys = ['text']
    datatype = torch.int64

    data = None
    if data_iterator is not None:
        try:
            data = next(data_iterator)
        except StopIteration:
            pass

    # 广播数据
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    if data_b is None:
        return None, None, None, None, None

    tokens = data_b['text'].long().cuda()
    tokens = tokens.transpose(0, 1).contiguous()
    labels = tokens[1:].contiguous()
    tokens = tokens[:-1].contiguous()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=tokenizer.eod,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss
    )
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

def dummy_provider(num_samples):
    # 这个函数现在只是个摆设，因为我们Patch了 builder
    return None

# ==========================================
# 4. 主程序 (含 Monkey Patch)
# ==========================================
if __name__ == "__main__":
    # -----------------------------------------------------------
    # 🟢 实施 Monkey Patch (劫持 Megatron 的数据加载逻辑)
    # -----------------------------------------------------------
    import megatron.training.training
    megatron.training.training.build_train_valid_test_data_iterators = fake_build_train_valid_test_data_iterators
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # 🟢 2. 手动初始化 Megatron 并修补缺失参数
    # -----------------------------------------------------------
    from megatron.training import initialize_megatron
    
    # 先手动初始化，传入和 pretrain 一样的默认参数
    initialize_megatron(args_defaults={'tokenizer_type': 'NullTokenizer'})
    
    # 获取全局 args 对象
    args = get_args()
    
    # 🩹 手动打补丁：补上缺失的控制属性
    # v0.14.0 在某些情况下漏掉了这些
    if not hasattr(args, 'do_train'):
        args.do_train = True
    if not hasattr(args, 'do_valid'):
        args.do_valid = True # 如果你想跑验证，就设为 True；否则 False
    if not hasattr(args, 'do_test'):
        args.do_test = False

    # 再次确认一下 vocab size 是否正确传进去了
    print_rank_0(f"DEBUG: Args check - do_train={args.do_train}, vocab_size={args.padded_vocab_size}")
    
    # -----------------------------------------------------------
    # 🟢 3. Init Function Patch (新增！防止 pretrain 再次初始化报错)
    # -----------------------------------------------------------
    # 定义一个空函数，假装自己是 initialize_megatron
    def fake_initialize(*args, **kwargs):
        print_rank_0(">> Skipping duplicate initialize_megatron call inside pretrain...")
        return

    # 【核心操作】把 training.py 里的初始化函数替换成这个空函数
    # 这样 pretrain 内部调用它时，什么都不会发生，也就不会报 "already initialized"
    megatron.training.training.initialize_megatron = fake_initialize

    # 初始化全局 Tokenizer
    global tokenizer
    tokenizer = MockTokenizer(151936) 

    pretrain(
        dummy_provider, # 传入也没用，会被上面的 Patch 拦截
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'NullTokenizer'}
    )