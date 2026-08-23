from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("LiquidAI/LFM2.5-350M-MLX-bf16")

response = generate(
    model,
    tokenizer,
    prompt="人工知能とは、",
    max_tokens=100,
    sampler=make_sampler(temp=0.1),
    verbose=True,
)
