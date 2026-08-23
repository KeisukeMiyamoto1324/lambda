from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("LiquidAI/LFM2.5-350M-MLX-4bit")

messages = [{"role": "user", "content": "人工知能とは何ですか？"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=100,
    sampler=make_sampler(temp=0.1),
    verbose=True,
)
