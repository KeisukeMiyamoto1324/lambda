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
        rotary_position_embedding: RotaryPositionEmbedding | None = None,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Split the model dimension into multiple heads so the same
        # attention module can be reused in a more general structure.
        # ---------------------------------------------------------
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rotary_position_embedding = rotary_position_embedding or RotaryPositionEmbedding(
            head_dim=self.head_dim,
        )

        # ---------------------------------------------------------
        # Project self-attention inputs into query, key, and value
        # spaces with one matrix multiplication.
        # ---------------------------------------------------------
        self.qkv_proj = nn.Linear(in_features=d_model, out_features=3 * d_model, bias=False)
        self.W_o = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Rearrange the last dimension into head count and head size
        # so attention can be computed independently per head.
        # ---------------------------------------------------------
        batch_size, seq_len, _ = x.size()
        reshaped = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
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
        q, k, v = [self._split_heads(projection) for projection in qkv.chunk(3, dim=-1)]

        # ---------------------------------------------------------
        # Apply rotary positions to queries and keys before the
        # attention scores are computed; values stay unrotated.
        # ---------------------------------------------------------
        q = self.rotary_position_embedding(q, position_ids=position_ids)
        k = self.rotary_position_embedding(k, position_ids=position_ids)

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
        q, current_k, current_v = [
            self._split_heads(projection)
            for projection in qkv.chunk(3, dim=-1)
        ]

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
        )

        # ---------------------------------------------------------
        # Return both the attention result and the updated cache for
        # this layer so the caller can feed the next token directly.
        # ---------------------------------------------------------
        merged_scores = self._merge_heads(attention_scores)
        return self.W_o(merged_scores), (k, v)
