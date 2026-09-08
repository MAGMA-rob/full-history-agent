"""Configuration only: importing this module never loads inference libraries."""
from typing import Literal
from magma_core.configs import BackendConfig
from pydantic import BaseModel, ConfigDict, Field


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    format: Literal["auto", "magma", "qwen", "gpt-oss"] = "auto"
    quantization: Literal["auto", "4bit", "8bit", "none"] = "auto"
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    max_new_tokens: int | None = Field(default=None, gt=0)
    attn_implementation: str | None = "sdpa"
    use_cache: bool = True
    enable_thinking: bool = False
    device_map: str = "auto"
    gpu_memory_limit: str | None = None
    allow_cpu_offload: bool = False
    offload_folder: str = "/tmp/agent-offload"
    chat_template: str | None = None
    output_style: Literal["qwen_format", "json"] = "qwen_format"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "0.0.0.0"
    port: int = Field(default=8888, ge=1, le=65535)
    prompt_log_dir: str | None = None
    coaching_backends: dict[str, BackendConfig] = Field(default_factory=dict)
    model: ModelSettings
