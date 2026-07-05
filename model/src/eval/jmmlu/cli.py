import argparse

from src.eval.shared.cli import build_eval_parser


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for JMMLU evaluation across native and
    # Hugging Face causal language models.
    # ---------------------------------------------------------
    parser = build_eval_parser()
    parser.add_argument("--subjects", nargs="*", default=None)
    return parser.parse_args()
