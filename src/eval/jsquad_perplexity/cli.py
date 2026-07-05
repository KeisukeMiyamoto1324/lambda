import argparse

from src.eval.shared.cli import build_eval_parser


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for JSQuAD context perplexity across
    # native and Hugging Face causal language models.
    # ---------------------------------------------------------
    return build_eval_parser().parse_args()
