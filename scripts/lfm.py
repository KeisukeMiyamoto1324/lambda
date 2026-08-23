import time

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

model_id = "LiquidAI/LFM2.5-350M-Base"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # dtype="bfloat16",
#   attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "カレーライスの作り方を解説します。"

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

start_time = time.perf_counter()
output = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.1,
    top_k=50,
    repetition_penalty=1.05,
    max_new_tokens=128,
    streamer=streamer,
)
elapsed_time = time.perf_counter() - start_time
generated_tokens = output.shape[-1] - inputs["input_ids"].shape[-1]
tokens_per_second = generated_tokens / elapsed_time

# ---------------------------------------------------------
# Display generation performance.
# ---------------------------------------------------------
print(f"\nGenerated tokens: {generated_tokens}")
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Generation speed: {tokens_per_second:.2f} tokens/s")
