<p align="center">
  <img src="assets/banner1.png" alt="Truly Open Japanese LLM development" width="100%">
</p>

# lambda

A small decoder-only Transformer project built with PyTorch and Lightning.

It includes tokenizer training, pretraining, midtraining, posttraining, and inference code.

## Setup

```bash
touch ~/.no_auto_tmux

sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y
sudo apt install nvtop

python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Usage

```bash
python3 src/tokenizer/train.py
python3 src/pretraining/train.py
python3 src/midtraining/train.py --model-path "models/lambda-160m"
python3 src/inference_base/inference.py --model-dir "models/lambda-160m-midtrained" --prompt "人工知能とは"
```

## PR: vast.ai
Vast.ai is a GPU cloud platform that lets you rent powerful GPUs from providers around the world. It is often more affordable than major cloud providers, with **NVIDIA H100 SXM GPUs available from around $1.54 per hour**. Check out Vast.ai for a lower-cost way to train lambda.

link: https://cloud.vast.ai/?ref_id=521936

## Test

```bash
python3 -m pytest
```
