import math
from transformers import PretrainedConfig


class WitchConfig(PretrainedConfig):
    model_type = "witch"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=None,
        kv_channels=None,
        ffn_hidden_size=4096,
        max_position_embeddings=4096,
        rotary_base=10000,
        rotary_pct=1.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        layer_norm_epsilon=1e-5,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.kv_channels = kv_channels if kv_channels is not None else hidden_size // num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rotary_base = rotary_base
        self.rotary_pct = rotary_pct
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        # derived
        self.head_dim = self.kv_channels
        self.num_q_per_kv = self.num_attention_heads // self.num_key_value_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
