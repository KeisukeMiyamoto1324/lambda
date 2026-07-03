import argparse


def parse_args() -> argparse.Namespace:
    # ---------------------------------------------------------
    # Define CLI arguments for JMMLU evaluation with native
    # PyTorch model artifacts used by this project.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    return parser.parse_args()
