from dataclasses import dataclass

from src.shared.tokenizer import ByteLevelBPE


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


def normalize_role(role: str) -> str:
    # ---------------------------------------------------------
    # Convert dataset-specific role names into the chat roles
    # represented by tokenizer special tokens.
    # ---------------------------------------------------------
    role_by_name = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    return role_by_name[role]


def get_role_token(tokenizer: ByteLevelBPE, role: str) -> str:
    # ---------------------------------------------------------
    # Resolve the special token string that marks each supported
    # chat role inside a serialized conversation.
    # ---------------------------------------------------------
    token_by_role = {
        "system": tokenizer.system_token,
        "user": tokenizer.user_token,
        "assistant": tokenizer.assistant_token,
    }
    return token_by_role[role]


def build_chat_input_ids(
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
    add_generation_prompt: bool,
) -> list[int]:
    # ---------------------------------------------------------
    # Serialize chat messages with the role and turn markers used
    # by instruction tuning.
    # ---------------------------------------------------------
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    end_of_turn_token_id = tokenizer.token_to_id(tokenizer.end_of_turn_token)
    input_ids = [bos_token_id]

    for message in messages:
        role = normalize_role(role=message.role)
        role_token_id = tokenizer.token_to_id(get_role_token(tokenizer=tokenizer, role=role))
        content_token_ids = tokenizer.tokenize(sentence=message.content)
        input_ids.extend([role_token_id, *content_token_ids, end_of_turn_token_id])

    # ---------------------------------------------------------
    # Add the assistant marker only when the next assistant reply
    # will be generated or scored.
    # ---------------------------------------------------------
    if add_generation_prompt:
        input_ids.append(tokenizer.token_to_id(tokenizer.assistant_token))

    return input_ids


def render_chat_generation_prompt(tokenizer: ByteLevelBPE, prompt: str) -> str:
    # ---------------------------------------------------------
    # Render one user prompt without BOS so native scoring can add
    # its standard BOS token before tokenization.
    # ---------------------------------------------------------
    return f"{tokenizer.user_token}{prompt}{tokenizer.end_of_turn_token}{tokenizer.assistant_token}"
