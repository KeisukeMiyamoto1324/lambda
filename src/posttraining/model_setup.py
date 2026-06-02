from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from src.pretraining.device_utils import resolve_device
from src.pretraining.transformer import DecoderOnlyTransformer
from src.tokenizer.tokenizer import ByteLevelBPE


DEFAULT_BASE_MODEL_ID = "MK0727/lambda-160m"


def download_base_model(base_model_id: str) -> Path:
    # ---------------------------------------------------------
    # Download the Hub snapshot so tokenizer and model artifacts
    # are available through the existing local loaders.
    # ---------------------------------------------------------
    return Path(snapshot_download(repo_id=base_model_id, repo_type="model"))


def build_tokenizer(base_model_dir: Path, output_path: Path) -> ByteLevelBPE:
    # ---------------------------------------------------------
    # Load the base tokenizer and save it beside the chat model
    # artifacts as a Hugging Face tokenizer directory.
    # ---------------------------------------------------------
    tokenizer = ByteLevelBPE.load(base_model_dir)
    tokenizer.save_pretrained(output_path)
    return tokenizer


def freeze_except_latter_half(model: DecoderOnlyTransformer) -> tuple[int, int]:
    # ---------------------------------------------------------
    # Freeze the full model first, then train only the latter half
    # of decoder blocks for partial fine tuning.
    # ---------------------------------------------------------
    trainable_layer_start = len(model.blocks) // 2
    trainable_layer_end = len(model.blocks)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in model.blocks[trainable_layer_start:trainable_layer_end]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    return trainable_layer_start, trainable_layer_end


def build_model_config(
    model: DecoderOnlyTransformer,
    learning_rate: float,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> dict[str, int | float]:
    # ---------------------------------------------------------
    # Build the compact config used by legacy and Hugging Face
    # artifact writers after posttraining completes.
    # ---------------------------------------------------------
    first_block = model.blocks[0]
    return {
        "max_len": model.pe.pe.size(dim=0),
        "d_model": model.we.embedding_dim,
        "num_layers": len(model.blocks),
        "num_heads": first_block.attention.num_heads,
        "d_ff": first_block.feed_forward.linear_1.out_features,
        "learning_rate": learning_rate,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
    }


def load_base_model(
    base_model_dir: Path,
    tokenizer: ByteLevelBPE,
    learning_rate: float,
    max_len: int,
    accelerator: str,
) -> tuple[DecoderOnlyTransformer, dict[str, int | float], int, int]:
    # ---------------------------------------------------------
    # Load the Hugging Face model, copy its weights into the local
    # Lightning model, and prepare it for partial fine tuning.
    # ---------------------------------------------------------
    hf_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        trust_remote_code=True,
        device_map=None,
    )
    model = DecoderOnlyTransformer(
        num_tokens=hf_model.config.vocab_size,
        d_model=hf_model.config.d_model,
        max_len=hf_model.config.max_len,
        num_layers=hf_model.config.num_layers,
        num_heads=hf_model.config.num_heads,
        d_ff=hf_model.config.d_ff,
        learning_rate=learning_rate,
        pad_token_id=tokenizer.token_to_id(tokenizer.pad_token),
    )
    model.load_state_dict(hf_model.transformer.state_dict())
    model = model.to(resolve_device())
    trainable_layer_start, trainable_layer_end = freeze_except_latter_half(model=model)
    model.learning_rate = learning_rate
    model.use_fused_optimizer = accelerator == "cuda"
    model.train()

    # ---------------------------------------------------------
    # Keep architecture metadata aligned with the downloaded model
    # and the tokenizer ids used by the SFT datasets.
    # ---------------------------------------------------------
    model_config = build_model_config(
        model=model,
        learning_rate=learning_rate,
        pad_token_id=tokenizer.token_to_id(tokenizer.pad_token),
        bos_token_id=tokenizer.token_to_id(tokenizer.bos_token),
        eos_token_id=tokenizer.token_to_id(tokenizer.eos_token),
    )
    return model, model_config, trainable_layer_start, trainable_layer_end
