from pathlib import Path

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from src.eval.shared.hf_scorer import TransformersChoiceScorer
from src.eval.shared.native_scorer import NativeChoiceScorer
from src.eval.shared.scorer_types import ChoiceScorer
from src.inference_base.generation import resolve_torch_dtype
from src.shared.device_utils import resolve_device
from src.shared.pytorch_artifacts import load_pytorch_model
from src.shared.tokenizer import ByteLevelBPE


def load_choice_scorer(
    model_source: str,
    backend: str,
    torch_dtype_name: str,
    trust_remote_code: bool,
) -> ChoiceScorer:
    # ---------------------------------------------------------
    # Resolve the backend from the model source and return one
    # scorer object with a shared evaluation interface.
    # ---------------------------------------------------------
    resolved_backend = resolve_backend(model_source=model_source, backend=backend)

    if resolved_backend == "native":
        return load_native_choice_scorer(
            model_source=model_source,
            torch_dtype_name=torch_dtype_name,
        )

    return load_transformers_choice_scorer(
        model_source=model_source,
        torch_dtype_name=torch_dtype_name,
        trust_remote_code=trust_remote_code,
    )


def resolve_backend(model_source: str, backend: str) -> str:
    # ---------------------------------------------------------
    # Auto-detect local native artifacts. Hub ids and other model
    # sources use the Transformers backend.
    # ---------------------------------------------------------
    if backend != "auto":
        return backend

    model_path = Path(model_source)

    if model_path.exists() and (model_path / "model.pth").exists() and (model_path / "model_config.json").exists():
        return "native"

    if is_local_model_path(model_source=model_source):
        raise FileNotFoundError(f"Native model artifacts were not found: {model_source}")

    return "hf"


def is_local_model_path(model_source: str) -> bool:
    # ---------------------------------------------------------
    # Treat common path-like model sources as local artifacts so
    # typos do not become confusing Hub requests.
    # ---------------------------------------------------------
    model_path = Path(model_source)
    return model_path.is_absolute() or model_source.startswith(("./", "../", "models/"))


def load_native_choice_scorer(model_source: str, torch_dtype_name: str) -> NativeChoiceScorer:
    # ---------------------------------------------------------
    # Load this project's PyTorch artifacts and tokenizer from a
    # local model directory.
    # ---------------------------------------------------------
    model_dir = Path(model_source)

    if not model_dir.exists():
        raise FileNotFoundError(f"Native model directory does not exist: {model_source}")

    tokenizer = ByteLevelBPE.load(model_dir)
    model, model_config = load_pytorch_model(
        model_dir=model_dir,
        vocab_size=tokenizer.get_vocab_size(),
    )
    device = resolve_device()
    torch_dtype = resolve_torch_dtype(torch_dtype=torch_dtype_name)
    model = model.to(device=device)

    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    model.eval()
    return NativeChoiceScorer(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=int(model_config["max_len"]),
        pad_token_id=tokenizer.token_to_id(tokenizer.pad_token),
        bos_token_id=tokenizer.token_to_id(tokenizer.bos_token),
        device=device,
        model_source=model_source,
        torch_dtype_name=torch_dtype_name,
    )


def load_transformers_choice_scorer(
    model_source: str,
    torch_dtype_name: str,
    trust_remote_code: bool,
) -> TransformersChoiceScorer:
    # ---------------------------------------------------------
    # Load a Hugging Face causal language model with Transformers
    # for external model comparison.
    # ---------------------------------------------------------
    device = resolve_device()
    torch_dtype = resolve_torch_dtype(torch_dtype=torch_dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device=device)
    model.eval()
    return TransformersChoiceScorer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        model_source=model_source,
        torch_dtype_name=torch_dtype_name,
    )
