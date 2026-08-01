import argparse


def build_eval_parser() -> argparse.ArgumentParser:
    # ---------------------------------------------------------
    # Build common CLI options for native and Hugging Face causal
    # language model evaluation scripts.
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backend", choices=["auto", "native", "hf"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    return parser


def add_prompt_format_argument(parser: argparse.ArgumentParser) -> None:
    # ---------------------------------------------------------
    # Add prompt formatting only to benchmarks that score an
    # assistant answer continuation.
    # ---------------------------------------------------------
    parser.add_argument("--prompt-format", choices=["auto", "base", "chat"], default="auto")
