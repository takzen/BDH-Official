# bdh.py
# Contains the full implementation of the BDH model architecture.
# Based on the reference code provided by the authors.
# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

@dataclasses.dataclass
class BDHConfig:
    """
    Configuration class for the BDH model.
    dataclass provides a clean way to define parameters.
    """
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256


def get_freqs(n, theta, dtype):
    """
    Calculates frequencies for Rotary Positional Encoding.
    """
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    """
    The core Attention module, implementing Linear Attention with RoPE.
    """
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        
        # The size of the conceptual space per head
        N = config.mlp_internal_dim_multiplier * D // nh
        
        # Frequencies are stored in a buffer, as they are not trainable parameters
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        """Helper function to calculate cos and sin from phases."""
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        """Applies Rotary Positional Encoding to a vector."""
        # Create a rotated version of the vector v
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        
        # Apply the rotation
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q # This attention mechanism assumes Q and K are the same
        
        _, _, T, _ = Q.size() # Get the sequence length T

        # Calculate phases for RoPE based on token position
        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        
        # Apply RoPE to Queries and Keys
        QR = self.rope(r_phases, Q)
        KR = QR # Since Q and K are the same

        # --- THIS IS THE LINEAR ATTENTION ---
        # Note the absence of softmax. This is the core of the efficient implementation.
        # .tril() applies the causal mask.
        scores = (QR @ KR.mT).tril(diagonal=-1)
        
        return scores @ V


class BDH(nn.Module):
    """
    The full BDH model.
    """
    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        nh = config.n_head
        D = config.n_embd
        
        # The size of the conceptual space per head
        N = D * config.mlp_internal_dim_multiplier // nh
        
        # --- Parameter Definitions ---
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        
        # --- Module Definitions ---
        self.attn = Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = nn.Embedding(config.vocab_size, D)
        self.drop = nn.Dropout(config.dropout)
        
        # --- Language Model Head ---
        self.lm_head = nn.Parameter(
            torch.zeros((D, config.vocab_size)).normal_(std=0.02)
        )
        # The original code has a "gate" which we can ignore for now for simplicity
        # self.lm_gate = nn.Parameter(torch.zeros((D, 1)).normal_(std=0.02))

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Custom weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # 1. Embedding
        x = self.embed(idx).unsqueeze(1) # (B, 1, T, D)
        x = self.ln(x)

        # 2. Processing through layers
        for _ in range(C.n_layer):
            # --- This is one block of the BDH architecture ---
            x_res = x # Store for residual connection
            
            # Project to conceptual space
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # (B, nh, T, N)

            # Attention in conceptual space
            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x, # Note: Values are the original, dense vectors
            )
            yKV = self.ln(yKV)

            # Modulation
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            # Project back to working space
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            y = self.ln(yMLP)
            
            # Apply residual connection
            x = self.ln(x_res + y)

        # 3. Final readout (Language Model Head)
        logits = x.view(B, T, D) @ self.lm_head
        
        # 4. Calculate loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # We don't need to crop context, the model handles long sequences
            logits, _ = self(idx)
            
            # Focus on the last token's logits
            logits = logits[:, -1, :] / temperature
            
            # Optional: Top-k sampling
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
                
            # Get probabilities and sample the next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append the new token
            idx = torch.cat((idx, idx_next), dim=1)
            
        self.train()
        return idx