"""Service layer for LLM operations using the unified interface."""

import logging
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from .base import LLMInterface
from .factory import LLMFactory

logger = logging.getLogger(__name__)

OutputType = TypeVar('OutputType', bound=BaseModel)


class LLMService:
	"""Service for managing LLM operations with unified interface."""
	
	def __init__(self, llm: BaseChatModel):
		self.adapter: LLMInterface = LLMFactory.create_adapter(llm)
		self._verify_connection()
	
	def _verify_connection(self) -> None:
		"""Verify the LLM connection during initialization."""
		if not self.adapter.verify_connection():
			raise ConnectionError("Failed to verify LLM connection")
	
	@property
	def capabilities(self):
		"""Get LLM capabilities."""
		return self.adapter.capabilities
	
	@property
	def model_name(self) -> str:
		"""Get the model name."""
		return self.adapter.model_name
	
	@property
	def library_name(self) -> str:
		"""Get the LLM library class name."""
		return self.adapter.library_name
	
	def supports_vision(self) -> bool:
		"""Check if the LLM supports vision."""
		return self.adapter.capabilities.supports_vision
	
	def supports_tool_calling(self) -> bool:
		"""Check if the LLM supports tool calling."""
		return self.adapter.capabilities.supports_tool_calling
	
	def get_tool_calling_method(self):
		"""Get the tool calling method being used."""
		return self.adapter.get_tool_calling_method()
	
	def set_tool_calling_method(self, method) -> None:
		"""Override the tool calling method."""
		self.adapter.set_tool_calling_method(method)
	
	async def get_structured_output(
		self,
		messages: list[BaseMessage],
		output_schema: type[OutputType],
		**kwargs
	) -> OutputType:
		"""Get structured output from the LLM."""
		return await self.adapter.get_structured_output(messages, output_schema, **kwargs)
	
	async def get_raw_output(self, messages: list[BaseMessage], **kwargs) -> str:
		"""Get raw text output from the LLM."""
		return await self.adapter.get_raw_output(messages, **kwargs)
	
	
