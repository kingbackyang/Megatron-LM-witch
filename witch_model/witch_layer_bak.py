import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb

class MegatronWitchAttention(MegatronModule):
    """
    WitchAttention custom implementation for Megatron-Core (TP Compatible).
    """

    def __init__(
        self,
        config,
        layer_number,
        attn_mask_type=AttnMaskType.causal,
        **kwargs
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        
        # --- 1. 基础配置 ---
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.kv_channels = config.kv_channels
        self.num_query_groups = config.num_query_groups 

        # --- 2. Tensor Parallel 设置 ---
        # 获取 TP 组的世界大小
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        
        # 计算当前 GPU 负责的头数
        self.num_heads_per_partition = self.num_attention_heads // tp_size
        
        # 当前 GPU 负责的 hidden_dim 大小 (用于 reshape)
        self.hidden_size_per_partition = self.num_heads_per_partition * self.kv_channels
        
        # --- 3. 投影层 (Projection Layers) ---
        
        # 🔴 关键修复：ColumnParallelLinear 需要传入【全局】的 output_size
        # Megatron 内部会自动将其除以 TP_SIZE。
        # 之前错误原因：传入了 self.hidden_size_per_partition，导致被除了两次。
        
        # Q Projection: 输出 Query + Gate (所以是 2 倍)
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,          # Input: 全局 Hidden
            config.hidden_size * 2,      # Output: 全局 Hidden * 2 (Megatron 会切分为 256*2)
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False
        )

        self.k_proj = ColumnParallelLinear(
            config.hidden_size,          # Input: 全局 Hidden
            config.hidden_size,          # Output: 全局 Hidden (Megatron 会切分为 256)
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False
        )

        self.v_proj = ColumnParallelLinear(
            config.hidden_size,          # Input: 全局 Hidden
            config.hidden_size,          # Output: 全局 Hidden (Megatron 会切分为 256)
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False
        )

        # Output Projection: 输入是切分的，输出是完整的 (这里会发生 All-Reduce)
        self.o_proj = RowParallelLinear(
            self.hidden_size, # Input: 本地切分的大小
            self.hidden_size,               # Output: 全局大小
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=False
        )

        # --- 4. Witch 组件 ---
        
        # Token Gate (放在 o_proj 之后以保证在完整的 Hidden State 上做卷积)
        mid_channels = max(4, self.hidden_size // 4)
        self.token_gate = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hidden_size,
                out_channels=mid_channels,
                kernel_size=5,
                padding=2,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=mid_channels,
                out_channels=self.hidden_size, 
                kernel_size=9,
                padding=4,
                bias=True,
            ),
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        **kwargs # 🔴 关键修复：吃掉 inference_context 等多余参数
    ):
        """
        hidden_states: [seq_len, batch_size, hidden_size]
        """
        
        # 获取维度信息
        # Megatron 标准输入布局: [Seq, Batch, Hidden]
        seq_len, batch_size, _ = hidden_states.shape

        # ===================================================================
        # Step 1: 投影 (Projections)
        # ===================================================================
        
        # q_mixed, k, v 的形状应该是: [seq_len, batch, hidden_per_partition]
        # (对于 q_mixed 是 hidden_per_partition * 2)
        q_mixed, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # 将 Q 分离为 Query 和 Gate Score
        # chunk 后，q 和 q_gate_score 的最后一维都是 hidden_per_partition
        q, q_gate_score = torch.chunk(q_mixed, 2, dim=-1)

        # Reshape for Attention
        # 🔴 关键修复：view 的顺序必须是 (Seq, Batch, Heads, Dim)
        # 使用 self.num_heads_per_partition (例如 4/4=1)
        q = q.view(seq_len, batch_size, self.num_heads_per_partition, self.kv_channels)
        k = k.view(seq_len, batch_size, self.num_heads_per_partition, self.kv_channels)
        v = v.view(seq_len, batch_size, self.num_heads_per_partition, self.kv_channels)
        q_gate_score = q_gate_score.view(seq_len, batch_size, self.num_heads_per_partition, self.kv_channels)

        # ===================================================================
        # Step 2: RoPE 位置编码
        # ===================================================================
        if rotary_pos_emb is not None:
            # 🔴 关键修复：传入 config=self.config
            q = apply_rotary_pos_emb(q, rotary_pos_emb, config=self.config)
            k = apply_rotary_pos_emb(k, rotary_pos_emb, config=self.config)

        # ===================================================================
        # Step 3: Attention 计算 (Flash/SDPA)
        # ===================================================================
        
        # 转置为 SDPA 需要的格式: [Batch, Heads, Seq, Dim]
        q_trans = q.permute(1, 2, 0, 3)
        k_trans = k.permute(1, 2, 0, 3)
        v_trans = v.permute(1, 2, 0, 3)

        # 使用 PyTorch 原生 SDPA (自动选择 FlashAttention)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            context = F.scaled_dot_product_attention(
                q_trans, k_trans, v_trans,
                attn_mask=None, # FlashAttn 内部处理 causal mask
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                is_causal=self.attn_mask_type == AttnMaskType.causal
            )
        
        # 转置回 Megatron 格式: [Seq, Batch, Heads, Dim]
        context = context.permute(2, 0, 1, 3)

        # ===================================================================
        # Step 4: 应用 Witch Query Gate (本地操作)
        # ===================================================================
        
        # context 和 q_gate_score 形状对齐，直接相乘
        context = context * torch.sigmoid(q_gate_score)

        # 展平 Heads 维度: [Seq, Batch, Hidden_Per_Partition]
        context = context.reshape(seq_len, batch_size, self.hidden_size_per_partition)

        # ===================================================================
        # Step 5: 输出投影 (All-Reduce)
        # ===================================================================
        
        # 输入是分片的，输出是完整的 (All-Reduce 发生在这里)
        # output_full: [Seq, Batch, Hidden_Size]
        output_full, _ = self.o_proj(context)

        # ===================================================================
        # Step 6: Token Gate (Global Conv1d)
        # ===================================================================
        
        # Conv1d 期望: [Batch, Channels, Seq_Len]
        # Megatron 数据: [Seq_Len, Batch, Hidden]
        
        # 1. 调整维度顺序
        conv_input = output_full.permute(1, 2, 0).contiguous()
        
        # 2. 卷积
        gate_feat = self.token_gate(conv_input)
        
        # 3. Sigmoid 激活
        gate_feat = torch.sigmoid(gate_feat)
        
        # 4. 调回顺序 [Seq, Batch, Hidden]
        gate_feat = gate_feat.permute(2, 0, 1)

        # 5. 应用 Gate
        final_output = output_full * gate_feat

        # 返回结果 (Megatron 期望返回 output, bias)
        return final_output, None