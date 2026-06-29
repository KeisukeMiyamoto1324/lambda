import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # RoPE rotates pairs of head features, so each attention head
        # must expose an even number of dimensions.
        # ---------------------------------------------------------
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary position embedding")

        # ---------------------------------------------------------
        # Keep inverse frequencies as deterministic state and cache
        # generated cos and sin tables outside saved checkpoints.
        # ---------------------------------------------------------
        embedding_index = torch.arange(start=0, end=head_dim, step=2).float()
        inv_freq = 1 / torch.tensor(10000.0) ** (embedding_index / head_dim)
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)

    def _extend_cache(self, position_count: int, device: torch.device) -> None:
        # ---------------------------------------------------------
        # Rebuild cached trig tables only when the requested maximum
        # position exceeds the current cache or the device changes.
        # ---------------------------------------------------------
        cache_is_ready = self.cos_cache.size(dim=0) >= position_count and self.cos_cache.device == device

        if cache_is_ready:
            return

        # ---------------------------------------------------------
        # Cache all positions up to the requested limit so later
        # shorter calls only gather from existing cos and sin tables.
        # ---------------------------------------------------------
        positions = torch.arange(
            start=0,
            end=position_count,
            device=device,
            dtype=self.inv_freq.dtype,
        )
        inv_freq = self.inv_freq.to(device=device)
        angles = positions.unsqueeze(-1) * inv_freq
        self.cos_cache = torch.cos(angles)
        self.sin_cache = torch.sin(angles)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        # ---------------------------------------------------------
        # Resolve explicit packed positions or contiguous positions
        # for regular full-sequence and cached inference paths.
        # ---------------------------------------------------------
        seq_len = x.size(dim=2)

        if position_ids is None:
            position_ids = torch.arange(
                start=position_offset,
                end=position_offset + seq_len,
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0)

        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        position_ids = position_ids.to(device=x.device, dtype=torch.long)

        # ---------------------------------------------------------
        # Gather cached cos and sin rows for explicit packed or
        # contiguous positions without recomputing trig each call.
        # ---------------------------------------------------------
        position_count = int(position_ids.max().item()) + 1
        self._extend_cache(position_count=position_count, device=x.device)
        cos = self.cos_cache[position_ids].to(dtype=x.dtype).unsqueeze(1)
        sin = self.sin_cache[position_ids].to(dtype=x.dtype).unsqueeze(1)

        # ---------------------------------------------------------
        # Rotate even and odd channels as complex-number pairs, then
        # flatten them back to the original head feature layout.
        # ---------------------------------------------------------
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        )
        return rotated.flatten(start_dim=-2)
