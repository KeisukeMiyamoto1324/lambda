import os
from pathlib import Path
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.shared.pytorch_artifacts import push_pytorch_model_artifacts


def main() -> None:
    # ---------------------------------------------------------
    # Load the Hugging Face token and instruction-tuning repository
    # name from .env, then select the completed chat model.
    # ---------------------------------------------------------
    load_dotenv()

    hf_token = os.environ["HF_TOKEN"]
    hf_repo = os.environ["HF_REPO_IT"]
    model_dir = Path("models/lambda-1-160m-it")

    # ---------------------------------------------------------
    # Push only PyTorch weights, model config, and tokenizer files.
    # Python source files and training outputs are skipped.
    # ---------------------------------------------------------
    push_pytorch_model_artifacts(
        output_path=model_dir,
        repo_id=hf_repo,
        private=True,
        commit_message="Upload lambda-1-160m instruction-tuned model",
        token=hf_token,
    )


if __name__ == "__main__":
    main()
