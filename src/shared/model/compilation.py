import torch

from src.shared.model.transformer import DecoderOnlyTransformer


COMPILE_MODE = "max-autotune-no-cudagraphs"


def compile_training_model(
    model: DecoderOnlyTransformer,
) -> DecoderOnlyTransformer:
    # ---------------------------------------------------------
    # Compile only the Transformer body so the Python loss-chunk loop
    # and vocabulary projections remain outside the compiled graph.
    # ---------------------------------------------------------
    model.forward_hidden = torch.compile(
        model.forward_hidden,
        mode=COMPILE_MODE,
        dynamic=False,
    )
    return model
