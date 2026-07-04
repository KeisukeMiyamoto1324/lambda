import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JSQUAD_DATASET_ID = "shunk031/JGLUE"
JSQUAD_CONFIG = "JSQuAD"
JSQUAD_SPLIT = "validation"
DEFAULT_CONTEXT_PATH = Path(__file__).resolve().parent / "jsquad_validation_contexts.json"


@dataclass(frozen=True)
class JSQuADContext:
    context_id: int
    context: str


def load_contexts(context_path: Path = DEFAULT_CONTEXT_PATH) -> list[JSQuADContext]:
    # ---------------------------------------------------------
    # Load pre-saved unique JSQuAD validation contexts. The file is
    # generated from JGLUE v1.2.0 before evaluation code runs.
    # ---------------------------------------------------------
    rows = json.loads(context_path.read_text(encoding="utf-8"))
    return [build_context(row=row) for row in rows]


def build_context(row: dict[str, Any]) -> JSQuADContext:
    # ---------------------------------------------------------
    # Convert one stored JSON row into the evaluator's stable
    # internal context shape.
    # ---------------------------------------------------------
    return JSQuADContext(
        context_id=int(row["context_id"]),
        context=str(row["context"]),
    )
