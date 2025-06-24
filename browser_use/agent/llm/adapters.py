"""Concrete LLM adapter implementations for different providers."""

import logging
from typing import TypeVar

try:
	from browser_use.agent.views import ToolCallingMethod
except ImportError:
	ToolCallingMethod = str

from .base import BaseLLMAdapter, LLMInterface
from .views import LLMCapabilities
from .utils import is_model_without_tool_support

logger = logging.getLogger(__name__)

OutputType = TypeVar('OutputType')

# Global registry for LLM adapters
_adapter_registry: list[type[LLMInterface]] = []


def llm_adapter(cls: type[LLMInterface]) -> type[LLMInterface]:
	"""Decorator to register an LLM adapter class."""
	if cls not in _adapter_registry:
		_adapter_registry.append(cls)
		logger.debug(f"Registered LLM adapter: {cls.__name__}")
	return cls


def get_registered_adapters() -> list[type[LLMInterface]]:
	"""Get all registered LLM adapter classes."""
	return _adapter_registry.copy()


@llm_adapter
class OpenAILLMAdapter(BaseLLMAdapter[OutputType]):
	"""Adapter for OpenAI ChatGPT models."""
	
	@classmethod
	def get_supported_llm_classes(cls) -> list[str]:
		"""Return list of LangChain class names this adapter supports."""
		return ['ChatOpenAI']
	
	@classmethod 
	def supports_model(cls, model_name: str) -> bool:
		"""Check if this adapter supports the given model name."""
		model_lower = model_name.lower()
		return any(pattern in model_lower for pattern in ['gpt-', 'davinci', 'turbo'])
	
	def _get_capabilities(self) -> LLMCapabilities:
		return LLMCapabilities(
			supports_vision=True,
			supports_tool_calling=True,
			preferred_tool_calling_method='function_calling',
			supports_structured_output=True,
			supports_streaming=True,
		)


@llm_adapter
class AzureOpenAILLMAdapter(BaseLLMAdapter[OutputType]):
	"""Adapter for Azure OpenAI models."""
	
	@classmethod
	def get_supported_llm_classes(cls) -> list[str]:
		"""Return list of LangChain class names this adapter supports."""
		return ['AzureChatOpenAI']
	
	@classmethod 
	def supports_model(cls, model_name: str) -> bool:
		"""Check if this adapter supports the given model name."""
		# Azure models typically have specific naming patterns
		return False  # Only match by class name for Azure
	
	def _get_capabilities(self) -> LLMCapabilities:
		# Azure OpenAI has different requirements for GPT-4
		tool_method = 'tools' if 'gpt-4-' in self.model_name.lower() else 'function_calling'
		
		return LLMCapabilities(
			supports_vision=True,
			supports_tool_calling=True,
			preferred_tool_calling_method=tool_method,
			supports_structured_output=True,
			supports_streaming=True,
		)


@llm_adapter
class AnthropicLLMAdapter(BaseLLMAdapter[OutputType]):
	"""Adapter for Anthropic Claude models."""
	
	@classmethod
	def get_supported_llm_classes(cls) -> list[str]:
		"""Return list of LangChain class names this adapter supports."""
		return ['ChatAnthropic', 'AnthropicChat']
	
	@classmethod 
	def supports_model(cls, model_name: str) -> bool:
		"""Check if this adapter supports the given model name."""
		model_lower = model_name.lower()
		return 'claude' in model_lower
	
	def _get_capabilities(self) -> LLMCapabilities:
		return LLMCapabilities(
			supports_vision=True,
			supports_tool_calling=True,
			preferred_tool_calling_method='tools',
			supports_structured_output=True,
			supports_streaming=True,
		)


@llm_adapter
class GoogleLLMAdapter(BaseLLMAdapter[OutputType]):
	"""Adapter for Google Gemini models."""
	
	@classmethod
	def get_supported_llm_classes(cls) -> list[str]:
		"""Return list of LangChain class names this adapter supports."""
		return ['ChatGoogleGenerativeAI']
	
	@classmethod 
	def supports_model(cls, model_name: str) -> bool:
		"""Check if this adapter supports the given model name."""
		model_lower = model_name.lower()
		return any(pattern in model_lower for pattern in ['gemini', 'palm'])
	
	def _get_capabilities(self) -> LLMCapabilities:
		return LLMCapabilities(
			supports_vision=True,
			supports_tool_calling=True,
			preferred_tool_calling_method=None,  # Uses native tool support
			supports_structured_output=True,
			supports_streaming=True,
		)


@llm_adapter
class DeepSeekLLMAdapter(BaseLLMAdapter[OutputType]):
	"""Adapter for DeepSeek models."""
	
	@classmethod
	def get_supported_llm_classes(cls) -> list[str]:
		"""Return list of LangChain class names this adapter supports."""
		return ['ChatDeepSeek']
	
	@classmethod 
	def supports_model(cls, model_name: str) -> bool:
		"""Check if this adapter supports the given model name."""
		model_lower = model_name.lower()
		return 'deepseek' in model_lower
	
	def _get_capabilities(self) -> LLMCapabilities:
		# DeepSeek models don't support vision yet
		supports_tools = not is_model_without_tool_support(self.model_name)
		
		return LLMCapabilities(
			supports_vision=False,
			supports_tool_calling=supports_tools,
			preferred_tool_calling_method='raw' if not supports_tools else 'function_calling',
			supports_structured_output=True,
			supports_streaming=True,
		)


@llm_adapter
class XAILLMAdapter(BaseLLMAdapter[OutputType]):
	"""Adapter for XAI Grok models."""
	
	@classmethod
	def get_supported_llm_classes(cls) -> list[str]:
		"""Return list of LangChain class names this adapter supports."""
		return []  # No dedicated LangChain class yet
	
	@classmethod 
	def supports_model(cls, model_name: str) -> bool:
		"""Check if this adapter supports the given model name."""
		model_lower = model_name.lower()
		return 'grok' in model_lower
	
	def _get_capabilities(self) -> LLMCapabilities:
		# XAI models don't support vision yet
		return LLMCapabilities(
			supports_vision=False,
			supports_tool_calling=True,
			preferred_tool_calling_method='function_calling',
			supports_structured_output=True,
			supports_streaming=True,
		)


class UniversalLLMAdapter(BaseLLMAdapter[OutputType]):
	"""Universal adapter for unknown or custom LLM implementations."""
	
	@classmethod
	def get_supported_llm_classes(cls) -> list[str]:
		"""Return list of LangChain class names this adapter supports."""
		return []  # Universal adapter doesn't match specific classes
	
	@classmethod 
	def supports_model(cls, model_name: str) -> bool:
		"""Check if this adapter supports the given model name."""
		return True  # Universal adapter supports everything as fallback
	
	def _get_capabilities(self) -> LLMCapabilities:
		"""Get capabilities with auto-detection for unknown models."""
		# Start with conservative defaults
		base_capabilities = LLMCapabilities(
			supports_vision=True,  # Will be validated later
			supports_tool_calling=True,
			preferred_tool_calling_method='auto',  # Will auto-detect
			supports_structured_output=True,
			supports_streaming=True,
		)
		
		# Override based on known patterns
		model_lower = self.model_name.lower()
		
		# Models known to not support tools
		if is_model_without_tool_support(self.model_name):
			base_capabilities.supports_tool_calling = False
			base_capabilities.preferred_tool_calling_method = 'raw'
		
		# Models known to not support vision
		if any(pattern in model_lower for pattern in ['deepseek', 'grok']):
			base_capabilities.supports_vision = False
		
		return base_capabilities