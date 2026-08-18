import torch
from torch._inductor import select_algorithm

from src.shared.model.transformer import DecoderOnlyTransformer


COMPILE_MODE = "max-autotune-no-cudagraphs"


def configure_compilation_output() -> None:
    # ---------------------------------------------------------
    # Hide per-kernel autotuning tables while preserving max-autotune
    # compilation, warnings, and compiler errors.
    # ---------------------------------------------------------
    select_algorithm.PRINT_AUTOTUNE = False


def compile_training_model(
    model: DecoderOnlyTransformer,
) -> DecoderOnlyTransformer:
    # ---------------------------------------------------------
    # Compile only the Transformer body so the fused vocabulary loss
    # remains outside the compiled graph and owns its Triton kernels.
    # ---------------------------------------------------------
    model.forward_hidden = torch.compile(
        model.forward_hidden,
        mode=COMPILE_MODE,
        dynamic=False,
    )
    return model
