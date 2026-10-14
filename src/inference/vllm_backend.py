"""Offline vLLM adapter for decoder-only local language models.

This module deliberately covers text generation only.  FactCC, mFACT,
MiniCheck, AlignScore, and the embedding retriever use their existing
Transformers/PyTorch adapters because they are classifier, encoder, or
encoder-decoder workloads rather than decoder-only generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


try:  # Jinja is a Transformers dependency when chat templates are used.
    from jinja2.exceptions import TemplateError as _TemplateError
except ImportError:  # pragma: no cover - only relevant to a broken runtime
    class _TemplateError(Exception):
        pass


def _render_chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    """Render one system/user exchange with the model's local chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError, _TemplateError):
            try:
                return apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, ValueError, _TemplateError):
                pass
    return f"{system_prompt}\n\n{user_prompt}\n\nJSON response:\n"


def _context_limit(config: Any) -> int | None:
    """Read the common Transformers context-limit fields."""
    for name in ("max_position_embeddings", "max_seq_len", "max_sequence_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


class VLLMTextGenerator:
    """Batch greedy generation through an already provisioned local vLLM."""

    backend_name = "local_vllm"

    def __init__(
        self,
        model_path: str | Path,
        *,
        base_model_path: str | Path | None = None,
        adapter_path: str | Path | None = None,
        dtype: str = "auto",
        max_input_tokens: int = 8192,
        max_new_tokens: int = 256,
        gpu_memory_utilization: float = 0.80,
        tensor_parallel_size: int = 1,
        max_model_length: int | None = None,
        seed: int = 42,
        allow_download: bool = False,
    ) -> None:
        if max_input_tokens < 1 or max_new_tokens < 1:
            raise ValueError("max_input_tokens and max_new_tokens must be positive")
        if not 0.0 < gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if max_model_length is not None and max_model_length < 1:
            raise ValueError("max_model_length must be positive when provided")
        if dtype not in {"auto", "float16", "bfloat16"}:
            raise ValueError("vLLM dtype must be auto, float16, or bfloat16")

        try:
            import vllm
        except ImportError as exc:  # pragma: no cover - depends on company runtime
            raise RuntimeError(
                "The vLLM inference backend was requested, but vLLM is not installed. "
                "Use the IT-provisioned vLLM environment or select "
                "--inference_backend transformers."
            ) from exc

        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - shared runtime failure
            raise RuntimeError(
                "The vLLM backend still requires the existing Transformers tokenizer/config runtime."
            ) from exc

        self.model_name = str(model_path)
        self.base_model_path = str(base_model_path or model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.max_model_length = max_model_length
        self.inference_backend = "vllm"
        local_files_only = not allow_download

        self.config = AutoConfig.from_pretrained(
            self.base_model_path,
            local_files_only=local_files_only,
        )
        if getattr(self.config, "is_encoder_decoder", False):
            raise ValueError(
                "The vLLM backend supports decoder-only language models such as Llama or Qwen. "
                "Use --inference_backend transformers for an encoder-decoder checkpoint."
            )
        self.context_limit = _context_limit(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        requested_length = max_model_length or (max_input_tokens + max_new_tokens)
        if self.context_limit is not None:
            requested_length = min(requested_length, self.context_limit)
        if requested_length < 1:
            raise ValueError("The model context length must be positive")
        self.effective_model_length = requested_length

        llm_kwargs: dict[str, Any] = {
            "model": self.base_model_path,
            "tokenizer": self.base_model_path,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": requested_length,
            "seed": seed,
        }
        if self.adapter_path:
            llm_kwargs["enable_lora"] = True

        self.llm = vllm.LLM(**llm_kwargs)
        self._sampling_params = vllm.SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
        )
        self._lora_request: Any | None = None
        if self.adapter_path:
            try:
                from vllm.lora.request import LoRARequest
            except ImportError as exc:  # pragma: no cover - vLLM version dependent
                raise RuntimeError(
                    "This vLLM installation does not expose LoRARequest; "
                    "use a merged FFT checkpoint or an IT-provisioned vLLM with LoRA support."
                ) from exc
            self._lora_request = LoRARequest(
                "vdt_local_adapter",
                1,
                self.adapter_path,
            )

    def _validate_prompt_lengths(self, prompts: Sequence[str]) -> None:
        if not prompts:
            return
        encoded = self.tokenizer(
            list(prompts),
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths = [len(item) for item in encoded["input_ids"]]
        too_long = [
            (index, length)
            for index, length in enumerate(lengths)
            if length > self.max_input_tokens
        ]
        if too_long:
            preview = ", ".join(f"{index}:{length}" for index, length in too_long[:8])
            raise ValueError(
                f"{len(too_long)} prompts exceed --max_input_tokens={self.max_input_tokens} ({preview}); "
                "no automatic truncation is performed"
            )
        if max(lengths) + self.max_new_tokens > self.effective_model_length:
            raise ValueError(
                "A prompt plus --max_new_tokens exceeds the vLLM model context; "
                "reduce the token limits or set a compatible --vllm_max_model_length"
            )

    def generate_prompts(self, prompts: Sequence[str]) -> list[str]:
        """Generate one continuation for each already-rendered prompt."""
        prompt_list = list(prompts)
        if not prompt_list:
            return []
        self._validate_prompt_lengths(prompt_list)
        kwargs: dict[str, Any] = {"use_tqdm": False}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outputs = self.llm.generate(
            prompt_list,
            self._sampling_params,
            **kwargs,
        )
        responses: list[str] = []
        for output in outputs:
            candidates = getattr(output, "outputs", None) or []
            responses.append(str(getattr(candidates[0], "text", "")) if candidates else "")
        if len(responses) != len(prompt_list):
            raise RuntimeError(
                f"vLLM returned {len(responses)} outputs for {len(prompt_list)} prompts"
            )
        return responses

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate_text_batch(system_prompt, [user_prompt])[0]

    def generate_text_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str]:
        prompts = [
            _render_chat_prompt(self.tokenizer, system_prompt, user_prompt)
            for user_prompt in user_prompts
        ]
        return self.generate_prompts(prompts)
