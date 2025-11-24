from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """
    Configuration for the Transformer language model.

    You can change these numbers to scale the model up or down:
      - d_model: hidden size
      - n_layers: number of Transformer blocks
      - n_heads: number of attention heads
      - d_ff: size of feedforward layer
      - max_seq_len: maximum context length
    """

    vocab_size: int          # must be set from tokenizer.vocab_size
    d_model: int = 256       # hidden size
    n_layers: int = 4        # number of Transformer blocks
    n_heads: int = 4         # number of attention heads
    d_ff: int = 1024         # feedforward inner dimension
    max_seq_len: int = 512   # maximum sequence length (context)
    dropout: float = 0.1     # dropout probability

    # You can add more hyperparameters later (e.g. layer norm eps, etc.)
    def to_dict(self):
        return asdict(self)
