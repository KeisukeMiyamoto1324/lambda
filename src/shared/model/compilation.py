import torch

from src.shared.model.transformer import DecoderOnlyTransformer


COMPILE_MODE = "max-autotune-no-cudagraphs"


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
