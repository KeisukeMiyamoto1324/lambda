import unittest

import torch
import torch.nn as nn

from src.inference_base.generation import stream_continuation_text
from src.shared.model.kv_cache import KeyValueCache


class FakeTokenizer:
    bos_token = "|<bos>|"
    eos_token = "|<eos>|"

    def token_to_id(self, token: str) -> int:
        # ---------------------------------------------------------
        # Return stable ids for prompt and stop tokens.
        # ---------------------------------------------------------
        token_ids = {
            self.bos_token: 1,
            self.eos_token: 2,
        }
        return token_ids[token]

    def tokenize(self, prompt: str) -> list[int]:
        # ---------------------------------------------------------
        # Keep prompt encoding fixed so generation calls are easy
        # to assert in the fake model.
        # ---------------------------------------------------------
        del prompt
        return [8, 9]

    def detokenize(self, token_ids: list[int]) -> str:
        # ---------------------------------------------------------
        # Decode cumulative generated ids so the streamer can emit
        # only the newly available text tail.
        # ---------------------------------------------------------
        text_by_token_id = {
            10: "Hel",
            11: "lo",
            12: "!",
        }
        return "".join(text_by_token_id[token_id] for token_id in token_ids)


class FakeModel(nn.Module):
    def __init__(self, next_token_ids: list[int]) -> None:
        super().__init__()

        # ---------------------------------------------------------
        # Keep one parameter for device lookup and emit fixed logits
        # for deterministic generation tests.
        # ---------------------------------------------------------
        self.probe = nn.Parameter(torch.zeros(1))
        self.next_token_ids = next_token_ids
        self.calls: list[list[int]] = []

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        past_key_values: KeyValueCache | None,
    ) -> tuple[torch.Tensor, KeyValueCache]:
        # ---------------------------------------------------------
        # Select the configured next token through greedy logits and
        # record the exact token ids passed into the cache path.
        # ---------------------------------------------------------
        del past_key_values
        self.calls.append([int(token_id) for token_id in token_ids[0].tolist()])
        token_id = self.next_token_ids[len(self.calls) - 1]
        logits = torch.zeros((1, token_ids.size(dim=1), 16), dtype=torch.float32)
        logits[0, -1, token_id] = 100.0
        return logits, []


class InferenceBaseTest(unittest.TestCase):
    def test_stream_continuation_text_yields_decoded_chunks(self) -> None:
        # ---------------------------------------------------------
        # Stream decoded text chunks as each token is generated while
        # using the cache path after the initial prompt.
        # ---------------------------------------------------------
        model = FakeModel(next_token_ids=[10, 11, 12])
        tokenizer = FakeTokenizer()
        chunks = list(
            stream_continuation_text(
                model=model,
                tokenizer=tokenizer,
                prompt="hello",
                max_new_tokens=3,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            )
        )

        self.assertEqual(chunks, ["Hel", "lo", "!"])
        self.assertEqual(model.calls, [[1, 8, 9], [10], [11]])

    def test_stream_continuation_text_stops_before_eos_chunk(self) -> None:
        # ---------------------------------------------------------
        # Stop generation at EOS and do not pass the stop token into
        # text decoding or terminal output.
        # ---------------------------------------------------------
        model = FakeModel(next_token_ids=[10, 2, 12])
        tokenizer = FakeTokenizer()
        chunks = list(
            stream_continuation_text(
                model=model,
                tokenizer=tokenizer,
                prompt="hello",
                max_new_tokens=8,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
            )
        )

        self.assertEqual(chunks, ["Hel"])
        self.assertEqual(model.calls, [[1, 8, 9], [10]])


if __name__ == "__main__":
    unittest.main()
