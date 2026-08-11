import unittest

import torch
import torch.nn as nn
from torchao.float8.float8_linear import Float8Linear

from src.shared.model.float8_training import convert_model_to_float8_training
from src.shared.model.float8_training import validate_float8_training_shapes
from src.shared.model.transformer import DecoderOnlyTransformer


class Float8TrainingTest(unittest.TestCase):
    def test_conversion_selects_large_projections_and_preserves_artifacts(self) -> None:
        # ---------------------------------------------------------
        # Convert only QKV and fused SwiGLU projections while keeping
        # state keys and tied vocabulary weights artifact-compatible.
        # ---------------------------------------------------------
        model = DecoderOnlyTransformer(
            num_tokens=32,
            d_model=16,
            max_len=8,
            num_layers=2,
            num_heads=2,
            d_ff=32,
            pad_token_id=0,
        )
        state_dict_keys = list(model.state_dict())
        embedding_weight = model.we.weight

        converted_model = convert_model_to_float8_training(model=model)

        self.assertIs(converted_model, model)
        self.assertEqual(list(converted_model.state_dict()), state_dict_keys)
        self.assertIs(converted_model.fc_layer.weight, embedding_weight)

        for block in converted_model.blocks:
            self.assertIsInstance(block.attention.qkv_proj, Float8Linear)
            self.assertIsInstance(block.feed_forward.gate_up_proj.input_proj, Float8Linear)
            self.assertIsInstance(block.feed_forward.gate_up_proj.output_proj, Float8Linear)
            self.assertIs(type(block.attention.W_o), nn.Linear)
            self.assertIs(type(block.feed_forward.down_proj.input_proj), nn.Linear)
            self.assertIs(type(block.feed_forward.down_proj.output_proj), nn.Linear)

        self.assertIs(type(converted_model.fc_layer), nn.Linear)

    def test_validation_rejects_unaligned_selected_projection(self) -> None:
        # ---------------------------------------------------------
        # Fail before conversion when a selected projection cannot
        # use native FP8 tensor-core matrix multiplication.
        # ---------------------------------------------------------
        model = DecoderOnlyTransformer(
            num_tokens=32,
            d_model=24,
            max_len=8,
            num_layers=1,
            num_heads=3,
            d_ff=32,
            pad_token_id=0,
        )

        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_float8_training_shapes(model=model)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9),
        "Native FP8 CUDA hardware is required",
    )
    def test_cuda_forward_backward_produces_finite_values(self) -> None:
        # ---------------------------------------------------------
        # Exercise native FP8 projection kernels under the BF16 outer
        # autocast used by Lightning and verify backward gradients.
        # ---------------------------------------------------------
        model = DecoderOnlyTransformer(
            num_tokens=32,
            d_model=16,
            max_len=8,
            num_layers=1,
            num_heads=2,
            d_ff=32,
            pad_token_id=0,
        ).cuda()
        model = convert_model_to_float8_training(model=model)
        input_tokens = torch.randint(1, 32, (2, 8), device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_tokens).sum()

        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(model.blocks[0].attention.qkv_proj.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
