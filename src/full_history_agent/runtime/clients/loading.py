from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from full_history_agent.config import ModelSettings
from .base import BaseModelClient


def detect_format(settings: ModelSettings) -> str:
    if settings.format != "auto":
        return settings.format
    tokenizer = AutoTokenizer.from_pretrained(settings.path)
    template = Path(settings.chat_template).read_text() if settings.chat_template else tokenizer.chat_template
    templates = " ".join(template.values()) if isinstance(template, dict) else (template or "")
    if "task_attributes" in templates and "permanent_rules" in templates:
        return "magma"
    config = AutoConfig.from_pretrained(settings.path)
    model_type = config.model_type.lower()
    if model_type == "gpt_oss":
        return "harmony"
    if model_type.startswith("qwen"):
        return "qwen"
    raise ValueError("Unknown model format; specify format=magma, qwen or harmony")


class CausalModelClient(BaseModelClient):
    """Local causal model with fixed placement for its entire lifetime."""

    def __init__(self, settings: ModelSettings, name: str) -> None:
        super().__init__(name=name, model_id=settings.path)
        self.use_cache = settings.use_cache
        self.enable_thinking = settings.enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(settings.path, padding_side="left")
        if settings.chat_template:
            self.tokenizer.chat_template = Path(settings.chat_template).read_text(encoding="utf-8")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = AutoConfig.from_pretrained(settings.path)
        self.max_new_tokens = settings.max_new_tokens
        native_quantization = getattr(config, "quantization_config", None)
        mode = settings.quantization
        if config.model_type == "gpt_oss" and mode != "auto":
            raise ValueError("GPT-OSS requires quantization=auto to preserve its checkpoint format")
        if native_quantization and mode != "auto":
            raise ValueError("Cannot override the quantization of an already quantized checkpoint")
        if mode == "auto":
            mode = "4bit" if settings.format == "qwen" and not native_quantization else "none"
        dtype = settings.dtype
        if dtype == "auto" and mode in {"4bit", "8bit"}:
            dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
        compute_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
        kwargs: dict[str, Any] = {
            "dtype": compute_dtype,
            "device_map": settings.device_map,
            "low_cpu_mem_usage": True,
        }
        if settings.attn_implementation:
            kwargs["attn_implementation"] = settings.attn_implementation
        if mode == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
                llm_int8_enable_fp32_cpu_offload=settings.allow_cpu_offload,
            )
        elif mode == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=settings.allow_cpu_offload,
            )
        if settings.gpu_memory_limit:
            kwargs["max_memory"] = {
                index: settings.gpu_memory_limit for index in range(torch.cuda.device_count())
            }
        if settings.allow_cpu_offload:
            kwargs["offload_folder"] = settings.offload_folder
        self.model = AutoModelForCausalLM.from_pretrained(settings.path, **kwargs)
        self.model.eval()

    @property
    def input_device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device
