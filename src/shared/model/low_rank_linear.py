import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Factor one large projection through a smaller rank while
        # keeping bias only on the final output projection.
        # ---------------------------------------------------------
        if rank <= 0:
            raise ValueError("rank must be greater than 0")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.input_proj = nn.Linear(
            in_features=in_features,
            out_features=rank,
            bias=False,
        )
        self.output_proj = nn.Linear(
            in_features=rank,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---------------------------------------------------------
        # Compress the input into the learned low-rank space before
        # expanding it to the requested output dimension.
        # ---------------------------------------------------------
        return self.output_proj(self.input_proj(x))
