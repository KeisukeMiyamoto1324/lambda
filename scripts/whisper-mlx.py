import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx_whisper
from mlx_whisper.transcribe import ModelHolder

MODEL_ID = "mlx-community/whisper-large-v3-turbo"

parser = argparse.ArgumentParser()
parser.add_argument("audio", type=Path)
args = parser.parse_args()

# ---------------------------------------------------------
# Load the model before measuring transcription speed.
# ---------------------------------------------------------
ModelHolder.get_model(MODEL_ID, mx.float16)
mx.reset_peak_memory()

start_time = time.perf_counter()
result = mlx_whisper.transcribe(
    str(args.audio),
    path_or_hf_repo=MODEL_ID,
    verbose=True,
)
elapsed_time = time.perf_counter() - start_time
generated_tokens = sum(len(segment["tokens"]) for segment in result["segments"])
tokens_per_second = generated_tokens / elapsed_time

# ---------------------------------------------------------
# Display transcription performance.
# ---------------------------------------------------------
print(f"Transcription: {generated_tokens} tokens, {tokens_per_second:.3f} tokens-per-sec")
print(f"Elapsed time: {elapsed_time:.3f} seconds")
print(f"Peak memory: {mx.get_peak_memory() / 1e9:.3f} GB")
