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
python3 src/inference_base/inference.py --prompt "人工知能とは"
```

## Test

```bash
python3 -m pytest
```
