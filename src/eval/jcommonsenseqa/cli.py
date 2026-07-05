import argparse

from src.eval.shared.cli import build_eval_parser


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for JCommonsenseQA evaluation across
    # native and Hugging Face causal language models.
    # ---------------------------------------------------------
    parser = build_eval_parser()
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    return parser.parse_args()
