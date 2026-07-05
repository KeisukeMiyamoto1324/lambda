import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------
# Add the project root to the import path so direct script
# execution can import the project packages consistently.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Load project environment variables before Hugging Face
# libraries need HF_TOKEN for authenticated requests.
# ---------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

from src.eval.jcommonsenseqa.cli import parse_args
from src.eval.jcommonsenseqa.runtime import run_evaluation


def main() -> None:
    # ---------------------------------------------------------
    # Keep this script as a small entrypoint. Implementation
    # details live in focused JCommonsenseQA modules.
    # ---------------------------------------------------------
    args = parse_args()
    run_evaluation(args=args)


if __name__ == "__main__":
    main()
