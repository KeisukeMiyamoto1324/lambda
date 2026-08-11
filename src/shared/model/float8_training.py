import torch.nn as nn
from torchao.float8 import Float8LinearConfig
from torchao.float8 import convert_to_float8_training

from src.shared.model.transformer import DecoderOnlyTransformer


FLOAT8_ALIGNMENT = 16
FLOAT8_RECIPE = "tensorwise"
TRAINING_PRECISION = "fp8"
FLOAT8_MODULE_SUFFIXES = (
    "attention.qkv_proj",
    "feed_forward.gate_up_proj.input_proj",
    "feed_forward.gate_up_proj.output_proj",
)


def is_float8_training_linear(module: nn.Module, fqn: str) -> bool:
    # ---------------------------------------------------------
    # Select only the large decoder projections that benefit from
    # tensorwise FP8 at this model scale.
    # ---------------------------------------------------------
    return isinstance(module, nn.Linear) and fqn.endswith(FLOAT8_MODULE_SUFFIXES)


def validate_float8_training_shapes(model: nn.Module) -> None:
    # ---------------------------------------------------------
    # Require hardware-aligned input and output dimensions for every
    # projection selected for native FP8 matrix multiplication.
    # ---------------------------------------------------------
    selected_modules = [
        (fqn, module)
        for fqn, module in model.named_modules()
        if is_float8_training_linear(module=module, fqn=fqn)
    ]

    for fqn, module in selected_modules:
        if module.in_features % FLOAT8_ALIGNMENT != 0 or module.out_features % FLOAT8_ALIGNMENT != 0:
            raise ValueError(
                f"FP8 linear dimensions must be divisible by {FLOAT8_ALIGNMENT}: {fqn}"
            )


def convert_model_to_float8_training(
    model: DecoderOnlyTransformer,
) -> DecoderOnlyTransformer:
    # ---------------------------------------------------------
    # Replace selected projections with TorchAO tensorwise FP8
    # modules while preserving their parameters and state keys.
    # ---------------------------------------------------------
    validate_float8_training_shapes(model=model)
    config = Float8LinearConfig.from_recipe_name(FLOAT8_RECIPE)
    convert_to_float8_training(
        model,
        config=config,
        module_filter_fn=is_float8_training_linear,
    )
    return model
