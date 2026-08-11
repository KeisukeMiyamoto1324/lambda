import torch

from src.shared.model.low_rank_linear import LowRankLinear
from src.shared.model.transformer import FeedForward


def test_low_rank_linear_preserves_requested_output_shape() -> None:
    # ---------------------------------------------------------
    # Project through the configured rank and preserve every leading
    # input dimension in the output tensor.
    # ---------------------------------------------------------
    layer = LowRankLinear(in_features=8, out_features=12, rank=4)
    inputs = torch.randn(2, 3, 8)

    outputs = layer(inputs)

    assert outputs.shape == (2, 3, 12)
    assert layer.input_proj.bias is None
    assert layer.output_proj.bias is None


def test_feed_forward_uses_half_model_width_for_both_ranks() -> None:
    # ---------------------------------------------------------
    # Keep one consistent bottleneck across the gate, value, and
    # down projections so every FFN block reduces parameters.
    # ---------------------------------------------------------
    feed_forward = FeedForward(d_model=960, d_ff=2560)

    assert feed_forward.gate_up_proj.rank == 480
    assert feed_forward.down_proj.rank == 480


def test_low_rank_feed_forward_reduces_default_model_parameters() -> None:
    # ---------------------------------------------------------
    # Verify the default FFN saves 2,764,800 parameters per block
    # compared with equivalent bias-free dense projections.
    # ---------------------------------------------------------
    feed_forward = FeedForward(d_model=960, d_ff=2560)
    parameter_count = sum(parameter.numel() for parameter in feed_forward.parameters())
    original_parameter_count = 3 * 960 * 2560

    assert original_parameter_count - parameter_count == 2_764_800
