<p align="center">
  <img src="assets/banner1.png" alt="Truly Open Japanese LLM development" width="100%">
</p>

# Lambda

A fully open Japanese language model project where the datasets, source code, and model weights are all publicly available.

## Overview

Lambda is an open project for learning how a Japanese language model is built. It covers the full process: tokenizer training, pretraining, midtraining, instruction tuning, inference, and evaluation.

The project uses streamed Japanese datasets and saves models in a Hugging Face-compatible format. The default base model has about 360 million parameters and is designed to be trained on [NVIDIA H100](https://cloud.vast.ai/?ref_id=521936) GPUs.

## PR: vast.ai
Vast.ai is a GPU cloud platform that lets you rent powerful GPUs from providers around the world. It is often more affordable than major cloud providers, with **NVIDIA H100 SXM GPUs available from around $1.54 per hour**. Check out Vast.ai for a lower-cost way to train lambda.

link: https://cloud.vast.ai/?ref_id=521936

## Models and Datasets

The trained 360M models are available on Hugging Face:

| Stage | Model |
|---|---|
| Pretraining | [lambda-1-360m-base](https://huggingface.co/KeisukeMiyamoto/lambda-1-360m-base) |
| Midtraining | [lambda-1-360m-mid](https://huggingface.co/KeisukeMiyamoto/lambda-1-360m-mid) |
| Posttraining | [lambda-1-360m-it](https://huggingface.co/KeisukeMiyamoto/lambda-1-360m-it) |

Each training stage uses the following dataset:

| Stage | Dataset |
|---|---|
| Pretraining | [lambda-corpus](https://huggingface.co/datasets/KeisukeMiyamoto/lambda-corpus) |
| Midtraining | [SyntheticTextbook-jp](https://huggingface.co/datasets/KeisukeMiyamoto/SyntheticTextbook-jp) |
| Posttraining | [SyntheticTalk-jp](https://huggingface.co/datasets/KeisukeMiyamoto/SyntheticTalk-jp) |

## Architecture

Lambda uses a decoder-only Transformer for next-token prediction. Its main components are grouped-query attention, rotary position embeddings, RMS normalization, SwiGLU feed-forward layers with low-rank projections, tied token embeddings, and a KV cache for faster generation.

The repository is organized by each stage of model development:

```text
src/
├── tokenizer/       Tokenizer training and encoding
├── pretraining/     Base model training
├── midtraining/     Continued training on textbook data
├── posttraining/    Instruction tuning for chat
├── inference_base/  Base model inference
├── inference_it/    Chat model inference
├── eval/            Japanese benchmark evaluation
└── shared/          Shared model and training components
```

## Requirements

- Python 3.12
- Internet access for Hugging Face models and datasets
- NVIDIA CUDA GPU with FP8 support for training; H100 is recommended

## Installation

First, clone the repository and move into the project directory. Then create a Python virtual environment and install dependencies.

```bash
git clone https://github.com/KeisukeMiyamoto1324/lambda.git
cd lambda
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Quick Start

The following commands cover the full workflow. Run them from the repository root after installation.

### 1. Tokenizer Training

Train a Japanese byte-level BPE tokenizer from the configured Hugging Face dataset.

```bash
python3 src/tokenizer/train.py --output-path models/tokenizer
```

### 2. Pretraining

Train the base model from scratch with the tokenizer created above.

```bash
python3 src/pretraining/train.py --tokenizer-path models/tokenizer --output-path models/lambda-360m
```

### 3. Midtraining

Continue training the base model on the Japanese textbook dataset.

```bash
python3 src/midtraining/train.py --model-path models/lambda-360m --output-path models/lambda-360m-midtrained
```

### 4. Posttraining

Continue from the midtrained model and train it with supervised instruction data.

```bash
python3 src/posttraining/train.py --model-path models/lambda-360m-midtrained --output-path models/lambda-360m-it
```

### 5. Inference

Generate a response with the instruction-tuned model created above.

```bash
python3 src/inference_it/inference.py --model-dir models/lambda-360m-it --prompt "人工知能とは何ですか？"
```
