import argparse


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for JCommonsenseQA evaluation across
    # native and Hugging Face causal language models.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backend", choices=["auto", "native", "hf"], default="auto")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    return parser.parse_args()
