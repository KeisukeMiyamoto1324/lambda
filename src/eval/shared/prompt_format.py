from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import Protocol

from src.shared.chat_template import render_chat_generation_prompt
from src.shared.tokenizer import ByteLevelBPE


PROMPT_FORMATS = ("auto", "base", "chat")


@dataclass(frozen=True)
class FormattedPrompt:
    text: str
    add_special_tokens: bool


class PromptFormatter(Protocol):
    name: str

    def format(self, prompt: str) -> FormattedPrompt:
        ...


@dataclass(frozen=True)
class BasePromptFormatter:
    name: str = "base"

    def format(self, prompt: str) -> FormattedPrompt:
        # ---------------------------------------------------------
        # Keep the existing causal language model prompt unchanged
        # and let the tokenizer add its normal special tokens.
        # ---------------------------------------------------------
        return FormattedPrompt(text=prompt, add_special_tokens=True)


@dataclass(frozen=True)
class NativeChatPromptFormatter:
    tokenizer: ByteLevelBPE
    name: str = "chat"

    def format(self, prompt: str) -> FormattedPrompt:
        # ---------------------------------------------------------
        # Wrap one benchmark prompt as a Lambda user turn followed
        # by the assistant generation marker.
        # ---------------------------------------------------------
        text = render_chat_generation_prompt(tokenizer=self.tokenizer, prompt=prompt)
        return FormattedPrompt(text=text, add_special_tokens=True)


@dataclass(frozen=True)
class HuggingFaceChatPromptFormatter:
    tokenizer: Any
    name: str = "chat"

    def format(self, prompt: str) -> FormattedPrompt:
        # ---------------------------------------------------------
        # Use the tokenizer-owned template and prevent a second BOS
        # or other model special token from being added later.
        # ---------------------------------------------------------
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        if not isinstance(text, str):
            raise TypeError("Hugging Face chat template must render text")

        return FormattedPrompt(text=text, add_special_tokens=False)


def build_native_prompt_formatter(
    prompt_format: str,
    tokenizer: ByteLevelBPE,
    model_config: Mapping[str, object],
) -> PromptFormatter:
    # ---------------------------------------------------------
    # Select raw or Lambda chat formatting from the CLI request
    # and the saved native model metadata.
    # ---------------------------------------------------------
    chat_template_version = model_config.get("chat_template_version")
    resolved_format = resolve_prompt_format(
        prompt_format=prompt_format,
        chat_template_available=chat_template_version is not None,
    )

    if resolved_format == "base":
        return BasePromptFormatter()

    if chat_template_version != 1:
        raise ValueError(f"Unsupported native chat template version: {chat_template_version}")

    return NativeChatPromptFormatter(tokenizer=tokenizer)


def build_hf_prompt_formatter(prompt_format: str, tokenizer: Any) -> PromptFormatter:
    # ---------------------------------------------------------
    # Select raw or tokenizer-owned chat formatting for a
    # Transformers model.
    # ---------------------------------------------------------
    chat_template_available = bool(getattr(tokenizer, "chat_template", None))
    resolved_format = resolve_prompt_format(
        prompt_format=prompt_format,
        chat_template_available=chat_template_available,
    )

    if resolved_format == "base":
        return BasePromptFormatter()

    return HuggingFaceChatPromptFormatter(tokenizer=tokenizer)


def resolve_prompt_format(prompt_format: str, chat_template_available: bool) -> str:
    # ---------------------------------------------------------
    # Resolve auto mode while rejecting an explicit chat request
    # when the selected model has no template metadata.
    # ---------------------------------------------------------
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"Unsupported prompt format: {prompt_format}")

    if prompt_format == "auto":
        return "chat" if chat_template_available else "base"

    if prompt_format == "chat" and not chat_template_available:
        raise ValueError("The selected model does not define a chat template")

    return prompt_format
