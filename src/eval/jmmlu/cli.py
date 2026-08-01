import argparse

from src.eval.shared.cli import build_eval_parser
from src.eval.shared.cli import add_prompt_format_argument


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for JMMLU evaluation across native and
    # Hugging Face causal language models.
    # ---------------------------------------------------------
    parser = build_eval_parser()
    add_prompt_format_argument(parser=parser)
    parser.add_argument("--subjects", nargs="*", default=None)
    return parser.parse_args()
