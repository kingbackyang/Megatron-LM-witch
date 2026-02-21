import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_witch import WitchConfig


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (..., seq, dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def build_rope(cache_seq_len, dim, base, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(cache_seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i , j -> i j", t, inv_freq)  # [seq, dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq, dim]
    cos = emb.cos()[None, None, :, :]  # [1,1,seq,dim]
    sin = emb.sin()[None, None, :, :]
    return cos, sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (self.weight * x).to(dtype=x.dtype)


class WitchAttention(nn.Module):
    def __init__(self, config: WitchConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.kv_channels = config.kv_channels
        self.q_size = self.num_heads * self.kv_channels
        self.kv_size = self.num_kv_heads * self.kv_channels
        self.num_q_per_kv = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.q_size * 2, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=True)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=True)

        mid_channels = max(4, self.hidden_size // 4)
        self.token_gate = nn.Sequential(
            nn.Conv1d(self.hidden_size, mid_channels, kernel_size=5, padding=2, bias=True),
            nn.GELU(),
            nn.Conv1d(mid_channels, self.hidden_size, kernel_size=9, padding=4, bias=True),
        )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_kv, slen, dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv, n_rep, slen, dim)
        return hidden_states.reshape(batch, num_kv * n_rep, slen, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        qkv = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, q_gate = torch.chunk(qkv, 2, dim=-1)

        q = q.view(bsz, seq_len, self.num_heads, self.kv_channels).transpose(1, 2)  # [B,H,S,D]
        q_gate = q_gate.view(bsz, seq_len, self.num_heads, self.kv_channels).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.kv_channels).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.kv_channels).transpose(1, 2)

        # Rotary
        q = q.transpose(1, 2)  # [B,S,H,D]
        k = k.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, rotary_cos, rotary_sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        if self.num_q_per_kv > 1:
            k = self.repeat_kv(k, self.num_q_per_kv)
            v = self.repeat_kv(v, self.num_q_per_kv)

        # SDPA expects [B, H, S, D]
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )  # [B,H,S,D]
        attn_output = attn_output * torch.sigmoid(q_gate)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)

        out = self.o_proj(attn_output)

        # token gate
        conv_input = out.transpose(1, 2).contiguous()  # [B, hidden, S]
        gate_feat = torch.sigmoid(self.token_gate(conv_input)).transpose(1, 2)
        out = out * gate_feat
        return out


class WitchMLP(nn.Module):
    def __init__(self, config: WitchConfig):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=True)
        self.act = nn.GELU()
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=True)

    def forward(self, x):
        x = self.act(self.dense_h_to_4h(x))
        return self.dense_4h_to_h(x)


class WitchBlock(nn.Module):
    def __init__(self, config: WitchConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.self_attention = WitchAttention(config)
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = WitchMLP(config)
        self.mlp_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attention(hidden_states, attention_mask, rotary_cos, rotary_sin)
        hidden_states = residual + self.post_attention_dropout(attn_output)

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp_dropout(self.mlp(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class WitchModel(PreTrainedModel):
    config_class = WitchConfig

    def __init__(self, config: WitchConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([WitchBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.rotary_dim = int(config.head_dim * config.rotary_pct)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.embed_tokens = embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache=None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
    ):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        # build rotary cache
        rotary_cos, rotary_sin = build_rope(
            seq_len, self.rotary_dim, self.config.rotary_base, device, dtype
        )

        # attention mask to sdpa mask: expect float with -inf on masked
        if attention_mask is not None:
            # attention_mask: [B,1,S,S] or [B,S]
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]  # [B,1,1,S]
            elif attention_mask.dim() == 4:
                mask = attention_mask
            else:
                raise ValueError("attention_mask must be 2D or 4D")
            attn_mask = (~mask.to(torch.bool)).to(dtype) * torch.finfo(dtype).min
        else:
            attn_mask = None

        all_hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attn_mask, rotary_cos, rotary_sin)

        hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            outputs = (hidden_states, None)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class WitchForCausalLM(PreTrainedModel):
    config_class = WitchConfig

    def __init__(self, config: WitchConfig):
        super().__init__(config)
        self.model = WitchModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.embed_tokens = embeddings

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states=False,
        output_attentions=False,
        return_dict=True,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            out = (logits,)
            if loss is not None:
                out = (loss,) + out
            if output_hidden_states:
                out += (outputs.hidden_states,)
            return out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
