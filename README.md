<p align="center">
  <img src="assets/banner1.png" alt="Truly Open Japanese LLM development" width="100%">
</p>

# lambda

A small decoder-only Transformer project built with PyTorch and Lightning.

It includes tokenizer training, pretraining, midtraining, posttraining, and inference code.

## Setup

```bash
pip3 install -r requirements.txt
```

## Usage

```bash
python3 src/tokenizer/train.py
python3 src/pretraining/train.py
python3 src/midtraining/train.py --model-path "models/lambda-160m"
python3 src/inference_base/inference.py --model-dir "models/lambda-160m-midtrained" --prompt "人工知能とは"
```

## PR: vast.ai
Vast.ai is a GPU cloud platform that lets you rent powerful GPUs from providers around the world. It is often more affordable than major cloud providers, helping reduce the cost of AI training, image generation, and other GPU-heavy tasks. Check out Vast.ai if you are looking for a lower-cost way to access high-performance GPUs.

link: https://cloud.vast.ai/?ref_id=521936

## Test

```bash
python3 -m pytest
```
