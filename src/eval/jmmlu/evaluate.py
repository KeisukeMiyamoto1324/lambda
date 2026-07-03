import sys
from pathlib import Path

# ---------------------------------------------------------
# Add the project root to the import path so direct script
# execution can import the project packages consistently.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.jmmlu.cli import parse_args
from src.eval.jmmlu.runtime import run_evaluation


def main() -> None:
    # ---------------------------------------------------------
    # Keep this script as a small entrypoint. Implementation
    # details live in focused JMMLU modules.
    # ---------------------------------------------------------
    args = parse_args()
    run_evaluation(args=args)


if __name__ == "__main__":
    main()
