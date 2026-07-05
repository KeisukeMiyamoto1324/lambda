from src.eval.shared.hf_scorer import TransformersChoiceScorer
from src.eval.shared.hf_scorer import resolve_hf_max_seq_len
from src.eval.shared.hf_scorer import resolve_hf_pad_token_id
from src.eval.shared.hf_scorer import score_hf_text
from src.eval.shared.hf_scorer import score_hf_text_chunk
from src.eval.shared.labeling import build_continuation_labels
from src.eval.shared.labeling import build_hf_labels
from src.eval.shared.labeling import encode_hf_text
from src.eval.shared.labeling import pad_token_row
from src.eval.shared.losses import compute_loss_sum
from src.eval.shared.losses import compute_row_losses
from src.eval.shared.losses import merge_text_scores
from src.eval.shared.native_scorer import NativeChoiceScorer
from src.eval.shared.native_scorer import score_native_answer_label
from src.eval.shared.native_scorer import score_native_continuation
from src.eval.shared.native_scorer import score_native_text
from src.eval.shared.native_scorer import score_native_text_chunk
from src.eval.shared.scorer_loader import is_local_model_path
from src.eval.shared.scorer_loader import load_choice_scorer
from src.eval.shared.scorer_loader import load_native_choice_scorer
from src.eval.shared.scorer_loader import load_transformers_choice_scorer
from src.eval.shared.scorer_loader import resolve_backend
from src.eval.shared.scorer_types import ChoiceScorer
from src.eval.shared.scorer_types import TextScore
from src.eval.shared.scorer_types import TextScorer


__all__ = [
    "ChoiceScorer",
    "NativeChoiceScorer",
    "TextScore",
    "TextScorer",
    "TransformersChoiceScorer",
    "build_continuation_labels",
    "build_hf_labels",
    "compute_loss_sum",
    "compute_row_losses",
    "encode_hf_text",
    "is_local_model_path",
    "load_choice_scorer",
    "load_native_choice_scorer",
    "load_transformers_choice_scorer",
    "merge_text_scores",
    "pad_token_row",
    "resolve_backend",
    "resolve_hf_max_seq_len",
    "resolve_hf_pad_token_id",
    "score_hf_text",
    "score_hf_text_chunk",
    "score_native_answer_label",
    "score_native_continuation",
    "score_native_text",
    "score_native_text_chunk",
]
