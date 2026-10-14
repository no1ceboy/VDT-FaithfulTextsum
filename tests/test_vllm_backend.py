from __future__ import annotations

import unittest
import types
from unittest.mock import patch

from src.inference.vllm_backend import VLLMTextGenerator, _render_chat_prompt


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return f"{messages[0]['content']}|{messages[1]['content']}"

    def __call__(self, prompts, **kwargs):
        del kwargs
        return {"input_ids": [[1, 2] for _ in prompts]}


class _Config:
    is_encoder_decoder = False
    max_position_embeddings = 64


class _FakeLLM:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs

    def generate(self, prompts, sampling_params, **kwargs):
        del sampling_params, kwargs
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text=f"response-{index}")]
            )
            for index, _prompt in enumerate(prompts)
        ]


class VLLMBackendTests(unittest.TestCase):
    def test_chat_prompt_uses_local_template(self) -> None:
        prompt = _render_chat_prompt(_Tokenizer(), "system", "user")
        self.assertEqual(prompt, "system|user")

    def test_missing_vllm_has_actionable_error(self) -> None:
        with patch.dict("sys.modules", {"vllm": None}):
            with self.assertRaisesRegex(RuntimeError, "vLLM.*not installed"):
                VLLMTextGenerator("models/example")

    def test_batch_generation_uses_vllm_and_preserves_order(self) -> None:
        fake_vllm = types.SimpleNamespace(
            LLM=_FakeLLM,
            SamplingParams=lambda **kwargs: kwargs,
        )
        tokenizer = _Tokenizer()
        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            with patch("transformers.AutoConfig.from_pretrained", return_value=_Config()):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
                    generator = VLLMTextGenerator(
                        "models/example",
                        max_input_tokens=8,
                        max_new_tokens=4,
                    )
                    responses = generator.generate_text_batch("system", ["one", "two"])
        self.assertEqual(responses, ["response-0", "response-1"])
        self.assertEqual(_FakeLLM.last_kwargs["max_model_len"], 12)


if __name__ == "__main__":
    unittest.main()
