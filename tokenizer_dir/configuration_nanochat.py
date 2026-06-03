from transformers import PretrainedConfig


class NanochatConfig(PretrainedConfig):
    model_type = "nanochat"
    attribute_map = {
        "hidden_size": "n_embd",
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "num_key_value_heads": "n_kv_head",
        "max_position_embeddings": "sequence_len",
    }

    def __init__(
        self,
        vocab_size=32768,
        padded_vocab_size=32768,
        sequence_len=2048,
        n_layer=24,
        n_head=12,
        n_kv_head=12,
        n_embd=1536,
        window_pattern="SSSL",
        # Standard HF aliases (accepted so configs remain loadable even if
        # written with generic field names by external tooling).
        hidden_size=None,
        num_hidden_layers=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        max_position_embeddings=None,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        **kwargs,
    ):
        if hidden_size is not None:
            n_embd = hidden_size
        if num_hidden_layers is not None:
            n_layer = num_hidden_layers
        if num_attention_heads is not None:
            n_head = num_attention_heads
        if num_key_value_heads is not None:
            n_kv_head = num_key_value_heads
        if max_position_embeddings is not None:
            sequence_len = max_position_embeddings

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.sequence_len = sequence_len
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern

        # Mirror common HF config keys for generation/cache utilities and
        # generic ecosystem tools that expect canonical names.
        self.hidden_size = self.n_embd
        self.num_hidden_layers = self.n_layer
        self.num_attention_heads = self.n_head
        self.num_key_value_heads = self.n_kv_head
        self.max_position_embeddings = self.sequence_len
        self.head_dim = self.n_embd // self.n_head
        self.intermediate_size = 4 * self.n_embd
        self.is_decoder = True
        self.use_cache = use_cache
        self.tie_word_embeddings = False
