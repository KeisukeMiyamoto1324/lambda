#!/usr/bin/env zsh
set -e

python3 src/eval/jsquad_perplexity/evaluate.py \
    --model /Users/keisukemiyamoto/Project/MyLLM/models/lambda-2-160m-base

python3 src/eval/jsquad_perplexity/evaluate.py \
    --model /Users/keisukemiyamoto/Project/MyLLM/models/lambda-2-160m-midtrain-base

python3 src/eval/jsquad_perplexity/evaluate.py \
    --model cyberagent/open-calm-small \
    --backend hf \
    --trust-remote-code

python3 src/eval/jsquad_perplexity/evaluate.py \
    --model cyberagent/open-calm-medium \
    --backend hf \
    --trust-remote-code

python3 src/eval/jsquad_perplexity/evaluate.py \
    --model llm-jp/llm-jp-3-150m \
    --backend hf \
    --trust-remote-code

python3 src/eval/jsquad_perplexity/evaluate.py \
    --model llm-jp/llm-jp-3-440m \
    --backend hf \
    --trust-remote-code

python3 src/eval/jsquad_perplexity/evaluate.py \
    --model Qwen/Qwen3-0.6B \
    --backend hf \
    --trust-remote-code
