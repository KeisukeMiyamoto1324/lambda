from dataclasses import dataclass

from src.shared.chat_template import ChatMessage
from src.shared.chat_template import get_role_token
from src.shared.chat_template import normalize_role
from src.shared.tokenizer import ByteLevelBPE


@dataclass(frozen=True)
class TokenizedChatExample:
    input_ids: list[int]
    labels: list[int]


def tokenize_chat_messages(
    tokenizer: ByteLevelBPE,
    messages: list[ChatMessage],
    max_len: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    end_of_turn_token_id: int,
) -> TokenizedChatExample:
    # ---------------------------------------------------------
    # Start each serialized conversation with BOS and keep its
    # target masked because it is structural context only.
    # ---------------------------------------------------------
    token_ids = [bos_token_id]
    target_mask = [False]

    # ---------------------------------------------------------
    # Append each role marker, content span, and end-of-turn marker
    # while enabling loss only on assistant content and turn end.
    # ---------------------------------------------------------
    for message in messages:
        role = normalize_role(message.role)
        role_token_id = tokenizer.token_to_id(get_role_token(tokenizer=tokenizer, role=role))
        content_token_ids = tokenizer.tokenize(sentence=message.content)
        is_assistant = role == "assistant"
        token_ids.extend([role_token_id, *content_token_ids, end_of_turn_token_id])
        target_mask.extend([False, *[is_assistant for _ in content_token_ids], is_assistant])

    # ---------------------------------------------------------
    # Close the sample with EOS and train on it only when it follows
    # an assistant answer, matching chat generation stop behavior.
    # ---------------------------------------------------------
    token_ids.append(eos_token_id)
    target_mask.append(target_mask[-1])

    # ---------------------------------------------------------
    # Convert the full token stream into shifted language-modeling
    # inputs and labels, masking every non-assistant next token.
    # ---------------------------------------------------------
    input_token_ids = token_ids[:-1][:max_len]
    shifted_token_ids = token_ids[1:][:max_len]
    shifted_target_mask = target_mask[1:][:max_len]
    label_token_ids = [
        token_id if is_target else pad_token_id
        for token_id, is_target in zip(shifted_token_ids, shifted_target_mask, strict=True)
    ]

    # ---------------------------------------------------------
    # Return unpadded streams so the dataset can combine several
    # independent conversations into one fixed-length sequence.
    # ---------------------------------------------------------
    return TokenizedChatExample(input_ids=input_token_ids, labels=label_token_ids)
