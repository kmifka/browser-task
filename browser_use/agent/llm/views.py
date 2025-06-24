"""LLM-related data models and view schemas."""

from typing import Any
from pydantic import BaseModel, Field

try:
	from browser_use.agent.views import ToolCallingMethod
except ImportError:
	# Fallback definition for standalone usage
	ToolCallingMethod = str


class LLMCapabilities(BaseModel):
	"""Capabilities and limitations of an LLM."""
	
	supports_vision: bool = True
	supports_tool_calling: bool = True
	preferred_tool_calling_method: ToolCallingMethod | None = 'function_calling'
	supports_structured_output: bool = True
	max_tokens: int | None = None
	supports_streaming: bool = True


class LLMConfig(BaseModel):
	"""Configuration for LLM operations."""
	
	max_retries: int = Field(default=3, ge=0)
	timeout_seconds: int | None = Field(default=None, ge=1)
	temperature: float | None = Field(default=None, ge=0.0, le=2.0)
	max_tokens: int | None = Field(default=None, ge=1)


class LLMResponse(BaseModel):
	"""Response from LLM operations."""
	
	content: str
	model_name: str
	tokens_used: int | None = None
	response_time_ms: int | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class LLMError(BaseModel):
	"""Error information from LLM operations."""
	
	error_type: str
	error_message: str
	model_name: str | None = None
	retry_count: int = 0
	is_retryable: bool = False