import sys
from pathlib import Path

# ---------------------------------------------------------
# Add the project root to the import path so direct script
# execution can import the project packages consistently.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_base.cli import parse_args
from src.inference_base.runtime import run_inference


DEFAULT_MODEL_ID = "KeisukeMiyamoto/lambda-1-160m-base"


def main() -> None:
    # ---------------------------------------------------------
    # Use the published Hub repository id as the default PyTorch
    # model source for inference.
    # ---------------------------------------------------------
    args = parse_args(default_model_dir=Path(DEFAULT_MODEL_ID))
    run_inference(args=args)


if __name__ == "__main__":
    main()
