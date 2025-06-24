"""Factory for creating appropriate LLM adapters."""

import logging
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel

from .adapters import (
	get_registered_adapters,
	UniversalLLMAdapter,
)
from .base import LLMInterface

logger = logging.getLogger(__name__)

OutputType = TypeVar('OutputType')


class LLMFactory:
	"""Factory for creating appropriate LLM adapters based on the LLM type."""
	
	@classmethod
	def create_adapter(cls, llm: BaseChatModel) -> LLMInterface:
		"""Create the appropriate LLM adapter for the given LLM instance."""
		llm_class_name = llm.__class__.__name__
		model_name = cls._extract_model_name(llm)
		
		# Get registered adapters (excluding UniversalLLMAdapter which is the fallback)
		available_adapters = [
			adapter for adapter in get_registered_adapters()
			if adapter != UniversalLLMAdapter
		]
		
		# First, check if any adapter supports this LangChain class
		for adapter_class in available_adapters:
			if llm_class_name in adapter_class.get_supported_llm_classes():
				logger.debug(f"Using {adapter_class.__name__} for {llm_class_name}")
				return adapter_class(llm)
		
		# Second, check if any adapter supports this model name
		if model_name:
			for adapter_class in available_adapters:
				if adapter_class.supports_model(model_name):
					logger.debug(f"Using {adapter_class.__name__} for model: {model_name}")
					return adapter_class(llm)
		
		# Fallback to universal adapter
		logger.debug(f"Using UniversalLLMAdapter for unknown LLM: {llm_class_name}/{model_name}")
		return UniversalLLMAdapter(llm)
	
	@staticmethod
	def _extract_model_name(llm: BaseChatModel) -> str:
		"""Extract model name from LLM instance."""
		if hasattr(llm, 'model_name') and llm.model_name:
			return llm.model_name
		elif hasattr(llm, 'model') and llm.model:
			return llm.model
		return 'Unknown'