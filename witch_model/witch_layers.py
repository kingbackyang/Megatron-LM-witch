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
from megatron.training import get_args

class MegatronWitchAttention(MegatronModule):
    """
    WitchAttention with GQA support and Custom Head Dim (128).
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
        args = get_args()
        
        # --- 1. 基础配置 & GQA 计算 ---
        self.hidden_size = config.hidden_size      # 1024
        self.num_attention_heads = config.num_attention_heads # 16
        self.kv_channels = config.kv_channels      # 128 (Head Dim)
        
        # GQA: 确定 KV 头数
        if args.group_query_attention:
            self.num_kv_heads = args.num_query_groups # 8
        else:
            self.num_kv_heads = self.num_attention_heads # 16

        # 计算 GQA 重复倍数 (16 / 8 = 2)
        self.num_q_per_kv = self.num_attention_heads // self.num_kv_heads

        # --- 2. Tensor Parallel 设置 ---
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        
        # 校验头数是否能被 TP 整除
        assert self.num_attention_heads % tp_size == 0
        assert self.num_kv_heads % tp_size == 0

        # 计算当前 GPU 负责的头数 (Local Heads)
        self.local_num_q_heads = self.num_attention_heads // tp_size
        self.local_num_kv_heads = self.num_kv_heads // tp_size
        
        # --- 3. 计算投影层的真实维度 ---
        # 注意：你的 config 里 16 * 128 = 2048，而 hidden_size 是 1024
        # 所以必须用 heads * kv_channels 来计算投影大小
        
        # Query 总维度 (全局)
        self.q_size = self.num_attention_heads * self.kv_channels
        # KV 总维度 (全局)
        self.kv_size = self.num_kv_heads * self.kv_channels

        # --- 4. 投影层 (Projection Layers) ---
        
        # Q Projection: 输出 Query + Gate
        # Query 和 Gate 的维度必须一致 (都是 16 头)，所以输出是 2 * q_size
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,          # Input: 1024
            self.q_size * 2,             # Output: 2048 * 2 = 4096 (全局)
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False
        )

        # K Projection
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,          # Input: 1024
            self.kv_size,                # Output: 1024 (8 * 128)
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False
        )

        # V Projection
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,          # Input: 1024
            self.kv_size,                # Output: 1024 (8 * 128)
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False
        )

        # Output Projection
        # 输入维度是 Attention 输出拼接后的维度 (也就是 Q 的维度 2048)
        # RowParallelLinear 会自动把 input_size 除以 TP_SIZE
        self.o_proj = RowParallelLinear(
            self.q_size,                 # Input: 2048 (全局)
            config.hidden_size,          # Output: 1024
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,      # 告诉 Megatron 输入已经是切分过的
            skip_bias_add=False
        )

        # --- 5. Witch 组件 (Token Gate) ---
        mid_channels = max(4, self.hidden_size // 4)
        self.token_gate = nn.Sequential(
            nn.Conv1d(self.hidden_size, mid_channels, kernel_size=5, padding=2, bias=True),
            nn.GELU(),
            nn.Conv1d(mid_channels, self.hidden_size, kernel_size=9, padding=4, bias=True),
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        **kwargs
    ):
        seq_len, batch_size, _ = hidden_states.shape

        # ===================================================================
        # Step 1: 投影 (Projections)
        # ===================================================================
        
        # q_mixed: [S, B, Local_Q_Dim * 2]
        q_mixed, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # 分离 Q 和 Q_Gate
        # chunk 后，q 和 q_gate_score 都是 [S, B, Local_Q_Dim]
        q, q_gate_score = torch.chunk(q_mixed, 2, dim=-1)

        # Reshape: [Seq, Batch, Local_Heads, Head_Dim]
        # 使用 self.local_num_q_heads (例如 16/TP) 和 self.local_num_kv_heads (8/TP)
        q = q.view(seq_len, batch_size, self.local_num_q_heads, self.kv_channels)
        q_gate_score = q_gate_score.view(seq_len, batch_size, self.local_num_q_heads, self.kv_channels)
        
        k = k.view(seq_len, batch_size, self.local_num_kv_heads, self.kv_channels)
        v = v.view(seq_len, batch_size, self.local_num_kv_heads, self.kv_channels)

        # ===================================================================
        # Step 2: RoPE 位置编码
        # ===================================================================
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb(q, rotary_pos_emb, config=self.config)
            k = apply_rotary_pos_emb(k, rotary_pos_emb, config=self.config)

        # ===================================================================
        # Step 3: GQA 广播 (Repeat KV)
        # ===================================================================
        # 目标: 将 K, V 从 8 头 扩展到 16 头，以便与 Q 对齐
        
        # 调整为 [Batch, Heads, Seq, Dim]
        q_trans = q.permute(1, 2, 0, 3)
        q_gate_trans = q_gate_score.permute(1, 2, 0, 3) # Witch Gate 也需要转置
        k_trans = k.permute(1, 2, 0, 3)
        v_trans = v.permute(1, 2, 0, 3)

        # 如果 KV 头数少于 Q 头数，进行复制
        if self.num_q_per_kv > 1:
            k_trans = self.repeat_kv(k_trans, self.num_q_per_kv)
            v_trans = self.repeat_kv(v_trans, self.num_q_per_kv)
            
        # 此时 k_trans 和 v_trans 的头数应该和 q_trans 一样了

        # ===================================================================
        # Step 4: Attention 计算
        # ===================================================================
        # 构造 mask：传入的 attention_mask 形状为 [B, 1, S, S]，可广播到 heads
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask
            if attn_mask.dim() == 4 and attn_mask.size(1) == 1:
                attn_mask = attn_mask  # [B,1,S,S]，后续广播到 heads
            attn_mask = attn_mask.to(torch.bool)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            context = F.scaled_dot_product_attention(
                q_trans, k_trans, v_trans,
                attn_mask=attn_mask,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                is_causal=self.attn_mask_type == AttnMaskType.causal
            )

        # ===================================================================
        # Step 5: Witch Query Gate (在 Heads 维度上操作)
        # ===================================================================
        
        # q_gate_trans 已经是 [Batch, Heads(16), Seq, Dim]
        # context 也是 [Batch, Heads(16), Seq, Dim]
        # 直接相乘即可
        context = context * torch.sigmoid(q_gate_trans)

        # 转置回 Megatron 格式: [Seq, Batch, Heads, Dim]
        context = context.permute(2, 0, 1, 3)
        
        # Flatten Heads: [Seq, Batch, Local_Q_Dim] -> 也就是 2048/TP
        context = context.reshape(seq_len, batch_size, -1)

        # ===================================================================
        # Step 6: 输出投影 (All-Reduce)
        # ===================================================================
        output_full, _ = self.o_proj(context)

        # ===================================================================
        # Step 7: Witch Token Gate (Conv1d)
        # ===================================================================
        conv_input = output_full.permute(1, 2, 0).contiguous()
        gate_feat = self.token_gate(conv_input)
        gate_feat = torch.sigmoid(gate_feat)
        gate_feat = gate_feat.permute(2, 0, 1)

        final_output = output_full * gate_feat

        return final_output, None

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        GQA Helper: copy KV heads to match Q heads.
        Input: [Batch, num_kv_heads, Seq, Dim]
        Output: [Batch, num_kv_heads * n_rep, Seq, Dim]
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
