import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.model.kv_cache import LayerKeyValueCache
from src.shared.model.position_encoding import RotaryPositionEmbedding


class Attention(nn.Module):
    def __init__(
        self,
        d_model: int = 2,
        num_heads: int = 1,
        num_kv_heads: int = 1,
        rotary_position_embedding: RotaryPositionEmbedding | None = None,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Split the model dimension into multiple heads so the same
        # attention module can be reused in a more general structure.
        # ---------------------------------------------------------
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.rotary_position_embedding = rotary_position_embedding or RotaryPositionEmbedding(
            head_dim=self.head_dim,
        )

        # ---------------------------------------------------------
        # Keep every query head while projecting fewer shared key and
        # value heads for grouped-query attention.
        # ---------------------------------------------------------
        self.qkv_proj = nn.Linear(
            in_features=d_model,
            out_features=d_model + 2 * self.kv_dim,
            bias=False,
        )
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        # ---------------------------------------------------------
        # Rearrange the last dimension into head count and head size
        # so attention can be computed independently per head.
        # ---------------------------------------------------------
        batch_size, seq_len, _ = x.size()
        reshaped = x.view(batch_size, seq_len, num_heads, self.head_dim)
        return reshaped.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Restore the tensor to the original model dimension after
        # per-head attention has been combined.
        # ---------------------------------------------------------
        batch_size, _, seq_len, _ = x.size()
        transposed = x.transpose(1, 2).contiguous()
        return transposed.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = False,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Create queries, keys, and values with one projection before
        # separating their attention heads.
        # ---------------------------------------------------------
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split((self.d_model, self.kv_dim, self.kv_dim), dim=-1)
        q = self._split_heads(q, num_heads=self.num_heads)
        k = self._split_heads(k, num_heads=self.num_kv_heads)
        v = self._split_heads(v, num_heads=self.num_kv_heads)

        # ---------------------------------------------------------
        # Apply rotary positions to queries and keys before the
        # attention scores are computed; values stay unrotated.
        # ---------------------------------------------------------
        q = self.rotary_position_embedding(q, position_ids=position_ids)
        k = self.rotary_position_embedding(k, position_ids=position_ids)

        # ---------------------------------------------------------
        # Add a singleton head axis so each packed sequence mask is
        # shared across all query heads.
        # ---------------------------------------------------------
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        # ---------------------------------------------------------
        # Use PyTorch's fused scaled dot-product attention so large
        # score and softmax tensors do not need to be materialized.
        # ---------------------------------------------------------
        attention_scores = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )

        # ---------------------------------------------------------
        # Merge the attended heads and project the result back into
        # the model dimension for the next layer.
        # ---------------------------------------------------------
        merged_scores = self._merge_heads(attention_scores)
        return self.W_o(merged_scores)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key_value: LayerKeyValueCache | None,
        is_causal: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, LayerKeyValueCache]:
        # ---------------------------------------------------------
        # Project the current tokens and append previous keys and
        # values so generation can avoid recomputing old states.
        # ---------------------------------------------------------
        qkv = self.qkv_proj(x)
        q, current_k, current_v = qkv.split(
            (self.d_model, self.kv_dim, self.kv_dim),
            dim=-1,
        )
        q = self._split_heads(q, num_heads=self.num_heads)
        current_k = self._split_heads(current_k, num_heads=self.num_kv_heads)
        current_v = self._split_heads(current_v, num_heads=self.num_kv_heads)

        # ---------------------------------------------------------
        # Rotate only the newly projected queries and keys. Cached
        # keys have already been stored with their final positions.
        # ---------------------------------------------------------
        q = self.rotary_position_embedding(q, position_offset=position_offset)
        current_k = self.rotary_position_embedding(current_k, position_offset=position_offset)

        k = current_k
        v = current_v

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, current_k), dim=2)
            v = torch.cat((past_v, current_v), dim=2)

        # ---------------------------------------------------------
        # Attend the current query positions over cached and current
        # keys with the fused scaled dot-product implementation.
        # ---------------------------------------------------------
        attention_scores = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )

        # ---------------------------------------------------------
        # Return both the attention result and the updated cache for
        # this layer so the caller can feed the next token directly.
        # ---------------------------------------------------------
        merged_scores = self._merge_heads(attention_scores)
        return self.W_o(merged_scores), (k, v)
