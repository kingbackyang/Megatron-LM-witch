import os
import torch
from functools import partial

# Megatron imports
from megatron.training import get_args, print_rank_0
from megatron.training import pretrain
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

# Import your custom module
from witch_layer import MegatronWitchAttention

def model_provider(pre_process=True, post_process=True):
    """
    Builds the model with custom WitchAttention.
    """
    args = get_args()

    print_rank_0("building GPT model with Custom Witch Attention ...")

    # 1. 获取默认的 GPT Layer Specification
    # 这个 Spec 包含了默认的 SelfAttention, MLP, LayerNorm 等定义
    transformer_layer_spec = get_gpt_layer_local_spec(
        num_experts=args.num_experts, 
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
    )

    # 2. 【关键步骤】替换 Self Attention 模块
    # 使用 ModuleSpec 告诉 Megatron 在构建 Transformer 层时使用你的类
    transformer_layer_spec.submodules.self_attention = ModuleSpec(
        module=MegatronWitchAttention,
        params={
            # 这里传递的参数会被 MegatronWitchAttention.__init__ 接收
            'attn_mask_type': args.attn_mask_type,
        },
    )

    # 3. 实例化 GPTModel
    model = GPTModel(
        config=args,
        transformer_layer_spec=transformer_layer_spec, # <--- 传入自定义 Spec
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=args.share_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent
    )

    return model

def get_batch(data_iterator):
    """Generate a batch."""
    # 这是标准的 GPT 数据处理逻辑
    keys = ['text']
    datatype = torch.int64

    # Broadcast data from rank 0 (standard Megatron pattern)
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
        
    # 这里省略了复杂的 broadcast 逻辑，通常直接调用 megatron.training.utils.get_batch_on_this_cp_rank
    # 为了简化，假设你使用标准的 dataset provider
    from megatron.training.utils import get_batch_on_this_cp_rank
    batch = get_batch_on_this_cp_rank(data_iterator)

    tokens = batch['text'].long()
    labels = tokens[:, 1:].contiguous()
    tokens = tokens[:, :-1].contiguous()

    # Get attention mask (Standard causal mask)
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss
    )

    return tokens, labels, loss_mask, attention_mask, position_ids

def forward_step(data_iterator, model):
    """Forward step."""
    # 获取数据
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)

    # 模型前向传播
    output_tensor = model(tokens, position_ids, attention_mask)

    # 计算 Loss
    def loss_func(loss_mask, output_tensor):
        losses = tensor_parallel.vocab_parallel_cross_entropy(
            output_tensor.float(), labels.float()
        )
        loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / loss_mask.sum()
        return loss, {'lm loss': loss}

    return output_tensor, loss_func

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """
    Placeholder for dataset provider.
    你需要根据你的数据格式（.bin/.idx）在这里加载数据。
    """
    print_rank_0('> building train, validation, and test datasets ...')
    # ... 实现你的数据加载逻辑 ...
    # return train_ds, val_ds, test_ds
    return None, None, None 

if __name__ == "__main__":
    
    # 注入自定义参数（如果 WitchConfig 有特殊的参数，在这里添加）
    def extra_args_provider(parser):
        group = parser.add_argument_group(title='witch_attention')
        # 如果你想把 conv kernel size 变成可配置的，可以在这里加
        # group.add_argument('--witch-kernel-size', type=int, default=5)
        return parser

    # 启动训练
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=extra_args_provider,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )