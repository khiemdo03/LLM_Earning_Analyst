from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention (causal), as used in GPT-style models.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len

        # Projection for query, key, value all at once: (d_model) -> (3 * d_model)
        self.qkv_proj = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (upper triangular) registered as buffer so it moves with device
        mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len))
        # Shape: (1, 1, T, T) for broadcasting (batch, heads, seq, seq)
        self.register_buffer("causal_mask", mask.view(1, 1, self.max_seq_len, self.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        returns: (batch_size, seq_len, d_model)
        """
        b, t, d = x.size()
        assert d == self.d_model, f"Expected d_model={self.d_model}, got {d}"
        assert t <= self.max_seq_len, f"Sequence length {t} > max_seq_len {self.max_seq_len}"

        # Project to q, k, v
        # qkv: (b, t, 3 * d_model)
        qkv = self.qkv_proj(x)
        # Split last dim into 3: (b, t, d_model) each
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head: (b, t, n_heads, d_head) -> (b, n_heads, t, d_head)
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention: (b, n_heads, t, t)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply causal mask: positions can only attend to <= current position
        # causal_mask: (1, 1, max_seq_len, max_seq_len)
        att_mask = self.causal_mask[:, :, :t, :t]  # (1, 1, t, t)
        att = att.masked_fill(att_mask == 0, float("-inf"))

        # Softmax along last dimension (keys)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values: (b, n_heads, t, d_head)
        y = att @ v

        # Rearrange back: (b, t, n_heads, d_head) -> (b, t, d_model)
        y = y.transpose(1, 2).contiguous().view(b, t, d)

        # Final projection
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y


class FeedForward(nn.Module):
    """
    Simple position-wise feedforward network: Linear -> GELU -> Linear -> Dropout
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Transformer block: pre-LN
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.ln_1(x))
        # Feedforward with residual
        x = x + self.ff(self.ln_2(x))
        return x


class TransformerModel(nn.Module):
    """
    GPT-style decoder-only Transformer for language modeling.

    Inputs:
      - token_ids: LongTensor of shape (batch_size, seq_len)

    Outputs:
      - logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # Positional embedding (learned)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Language modeling head (tied weights optional later)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights in a somewhat standard way.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        input_ids: LongTensor (batch_size, seq_len)
        returns: logits (batch_size, seq_len, vocab_size)
        """
        b, t = input_ids.size()
        if t > self.max_seq_len:
            raise ValueError(f"Sequence length {t} > max_seq_len {self.max_seq_len}")

        device = input_ids.device

        # Token + positional embeddings
        tok_emb = self.tok_emb(input_ids)  # (b, t, d_model)

        # Positions: 0..t-1
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        pos_emb = self.pos_emb(pos)  # (1, t, d_model)

        x = tok_emb + pos_emb  # (b, t, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)  # (b, t, d_model)

        # Project to vocab
        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            input_ids: starting tokens (b, t_start)
            max_new_tokens: how many tokens to generate
            temperature: >0, scales logits (higher = more random)
            top_k: if set, keep only top_k logits before softmax

        Returns:
            output_ids: (b, t_start + max_new_tokens)
        """
        self.eval()

        out = input_ids
        for _ in range(max_new_tokens):
            # If sequence is longer than max_seq_len, crop from the left
            if out.size(1) > self.max_seq_len:
                out_cond = out[:, -self.max_seq_len :]
            else:
                out_cond = out

            # Forward pass to get logits for the last position
            logits = self(out_cond)  # (b, t, vocab_size)
            logits = logits[:, -1, :]  # (b, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Optionally top-k filter
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution
            next_ids = torch.multinomial(probs, num_samples=1)  # (b, 1)

            # Append to sequence
            out = torch.cat([out, next_ids], dim=1)

        return out

    def num_parameters(self, trainable_only: bool = True) -> int:
        """
        Return the total number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def to_dict(self) -> dict:
        """
        Return the config as a dict, useful for saving.
        """
        return asdict(self.config)
