"""Base LLM interface and common abstractions."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

try:
	from browser_use.agent.views import ToolCallingMethod
	from browser_use.exceptions import LLMException
except ImportError:
	# Fallback definitions for standalone usage
	ToolCallingMethod = str
	LLMException = Exception

from .utils import extract_json_from_model_output, clean_response_content
from .views import LLMCapabilities

logger = logging.getLogger(__name__)

OutputType = TypeVar('OutputType', bound=BaseModel)


class LLMInterface(ABC, Generic[OutputType]):
	"""Abstract interface for LLM implementations with unified capabilities."""
	
	def __init__(self, llm: BaseChatModel):
		self.llm = llm
		self.model_name = self._extract_model_name()
		self.capabilities = self._get_capabilities()
		self._verified = False
		self._tool_calling_method: ToolCallingMethod | None = None
		
	def _extract_model_name(self) -> str:
		"""Extract model name from LLM instance."""
		if hasattr(self.llm, 'model_name') and self.llm.model_name:
			return self.llm.model_name
		elif hasattr(self.llm, 'model') and self.llm.model:
			return self.llm.model
		return 'Unknown'
	
	@abstractmethod
	def _get_capabilities(self) -> LLMCapabilities:
		"""Get the capabilities for this LLM implementation."""
		pass
	
	@abstractmethod
	def _test_tool_calling_method(self, method: ToolCallingMethod) -> bool:
		"""Test if a specific tool calling method works."""
		pass
		
	def _detect_tool_calling_method(self) -> ToolCallingMethod | None:
		"""Auto-detect the best tool calling method."""
		if not self.capabilities.supports_tool_calling:
			return 'raw'
			
		# Try methods in order of preference
		methods_to_try = [
			'function_calling',
			'tools', 
			'json_mode',
			'raw'
		]
		
		# Start with known good method if available
		if self.capabilities.preferred_tool_calling_method:
			methods_to_try.remove(self.capabilities.preferred_tool_calling_method)
			methods_to_try.insert(0, self.capabilities.preferred_tool_calling_method)
		
		for method in methods_to_try:
			try:
				if self._test_tool_calling_method(method):
					logger.debug(f"Selected tool calling method: {method}")
					return method
			except Exception as e:
				logger.debug(f"Tool calling method {method} failed: {e}")
				continue
				
		raise ConnectionError('Failed to find working tool calling method')
	
	def get_tool_calling_method(self) -> ToolCallingMethod | None:
		"""Get the tool calling method, detecting it if needed."""
		if self._tool_calling_method is None:
			self._tool_calling_method = self._detect_tool_calling_method()
		return self._tool_calling_method
	
	def set_tool_calling_method(self, method: ToolCallingMethod | None) -> None:
		"""Override the tool calling method."""
		self._tool_calling_method = method
	
	@abstractmethod
	async def get_structured_output(
		self,
		messages: list[BaseMessage],
		output_schema: type[OutputType],
		**kwargs
	) -> OutputType:
		"""Get structured output from the LLM.
		
		For raw method, actions will be extracted from output_schema and added to context.
		For structured methods, output_schema is used directly for tool calling.
		"""
		pass
	
	@abstractmethod
	async def get_raw_output(self, messages: list[BaseMessage], **kwargs) -> str:
		"""Get raw text output from the LLM."""
		pass
	
	
	@classmethod
	@abstractmethod
	def get_supported_llm_classes(cls) -> list[str]:
		"""Return list of LangChain class names this adapter supports."""
		pass
	
	@classmethod
	@abstractmethod
	def supports_model(cls, model_name: str) -> bool:
		"""Check if this adapter supports the given model name."""
		pass
	
	def verify_connection(self) -> bool:
		"""Verify the LLM connection works."""
		if self._verified:
			return True
			
		try:
			# Simple test to verify API key and connection
			test_msg = [HumanMessage(content="Say 'OK'")]
			response = self.llm.invoke(test_msg)
			self._verified = bool(response and hasattr(response, 'content'))
			return self._verified
		except Exception as e:
			logger.error(f"LLM connection verification failed: {e}")
			return False
	
	@property
	def library_name(self) -> str:
		"""Get the LLM library class name."""
		return self.llm.__class__.__name__
	
	def _convert_messages_if_needed(self, messages: list[BaseMessage]) -> list[BaseMessage]:
		"""Convert messages internally if this model needs it."""
		from .utils import convert_input_messages
		return convert_input_messages(messages, self.model_name)


class BaseLLMAdapter(LLMInterface[OutputType], ABC):
	"""Base implementation with common functionality."""
	
	def _test_tool_calling_method(self, method: ToolCallingMethod) -> bool:
		"""Test if a specific tool calling method works."""
		try:
			# Simple test schema for verification
			class TestResponse(BaseModel):
				answer: str
				
			test_question = "What is the capital of France? Respond with just the city name."
			test_messages = [HumanMessage(content=test_question)]
			
			if method == 'raw':
				response = self.llm.invoke(test_messages)
				content = str(response.content).strip().lower()
				return 'paris' in content
			else:
				structured_llm = self.llm.with_structured_output(
					TestResponse, 
					include_raw=True, 
					method=method
				)
				response = structured_llm.invoke(test_messages)
				
				if hasattr(response, 'parsed') and response.parsed:
					return 'paris' in response.parsed.answer.lower()
				elif isinstance(response, dict) and 'parsed' in response:
					parsed = response['parsed']
					return parsed and 'paris' in parsed.answer.lower()
				
				return False
				
		except Exception as e:
			logger.debug(f"Tool calling test failed for method {method}: {e}")
			return False
	
	async def get_raw_output(self, messages: list[BaseMessage], **kwargs) -> str:
		"""Get raw text output from the LLM."""
		try:
			# Convert messages internally if needed
			converted_messages = self._convert_messages_if_needed(messages)
			response = await self.llm.ainvoke(converted_messages)
			return str(response.content)
		except Exception as e:
			logger.error(f"Failed to get raw output: {e}")
			raise LLMException(401, "LLM API call failed") from e
	
	async def get_structured_output(
		self,
		messages: list[BaseMessage], 
		output_schema: type[OutputType],
		**kwargs
	) -> OutputType:
		"""Get structured output from the LLM using the best available method."""
		# Convert messages internally if needed
		converted_messages = self._convert_messages_if_needed(messages)
		
		tool_method = self.get_tool_calling_method()
		
		if tool_method == 'raw':
			# For raw method, add schema format as system message
			converted_messages = self._add_schema_as_system_message(converted_messages, output_schema)
			return await self._get_structured_output_raw(converted_messages, output_schema, **kwargs)
		elif tool_method is None:
			return await self._get_structured_output_native(converted_messages, output_schema, **kwargs)
		else:
			return await self._get_structured_output_tools(converted_messages, output_schema, tool_method, **kwargs)
	
	async def _get_structured_output_raw(
		self,
		messages: list[BaseMessage],
		output_schema: type[OutputType], 
		**kwargs
	) -> OutputType:
		"""Get structured output using raw JSON parsing."""
		try:
			raw_output = await self.get_raw_output(messages, **kwargs)
			# Clean up response content (remove <think> tags for reasoning models)
			cleaned_output = clean_response_content(raw_output)
			parsed_json = extract_json_from_model_output(cleaned_output)
			return output_schema(**parsed_json)
		except Exception as e:
			logger.error(f"Failed to parse raw structured output: {e}")
			raise LLMException(401, "Failed to parse LLM response") from e
	
	async def _get_structured_output_native(
		self,
		messages: list[BaseMessage],
		output_schema: type[OutputType],
		**kwargs
	) -> OutputType:
		"""Get structured output using native LangChain structured output."""
		try:
			structured_llm = self.llm.with_structured_output(output_schema, include_raw=True)
			response = await structured_llm.ainvoke(messages)
			
			if isinstance(response, dict):
				parsed = response.get('parsed')
				if parsed and isinstance(parsed, output_schema):
					return parsed
			elif hasattr(response, 'parsed') and response.parsed:
				return response.parsed
				
			# Fallback to raw parsing if structured parsing failed
			if isinstance(response, dict) and 'raw' in response:
				raw_content = str(response['raw'].content)
			else:
				raw_content = str(getattr(response, 'content', response))
				
			cleaned_content = clean_response_content(raw_content)
			parsed_json = extract_json_from_model_output(cleaned_content)
			return output_schema(**parsed_json)
			
		except Exception as e:
			logger.error(f"Failed to get native structured output: {e}")
			raise LLMException(401, "LLM API call failed") from e
	
	async def _get_structured_output_tools(
		self,
		messages: list[BaseMessage],
		output_schema: type[OutputType],
		method: ToolCallingMethod,
		**kwargs
	) -> OutputType:
		"""Get structured output using tool calling methods."""
		try:
			structured_llm = self.llm.with_structured_output(
				output_schema, 
				include_raw=True, 
				method=method
			)
			response = await structured_llm.ainvoke(messages)
			
			# Handle parsing errors and tool calls
			if isinstance(response, dict):
				if response.get('parsing_error') and 'raw' in response:
					return self._handle_tool_call_response(response, output_schema)
				elif 'parsed' in response and response['parsed']:
					return response['parsed']
			elif hasattr(response, 'parsed') and response.parsed:
				return response.parsed
				
			# Fallback to raw parsing
			raw_content = self._extract_raw_content(response)
			cleaned_content = clean_response_content(raw_content)
			parsed_json = extract_json_from_model_output(cleaned_content)
			return output_schema(**parsed_json)
			
		except Exception as e:
			logger.error(f"Failed to get tool-based structured output: {e}")
			raise LLMException(401, "LLM API call failed") from e
	
	def _handle_tool_call_response(self, response: dict, output_schema: type[OutputType]) -> OutputType:
		"""Handle tool call responses when parsing errors occur."""
		raw_msg = response['raw']
		if hasattr(raw_msg, 'tool_calls') and raw_msg.tool_calls:
			tool_call = raw_msg.tool_calls[0]
			
			# Create a basic structure that matches the expected schema
			tool_call_args = tool_call.get('args', {})
			
			# Try to map tool call to output schema
			if hasattr(output_schema, 'model_fields'):
				# For pydantic models, try to construct from tool call args
				return output_schema(**tool_call_args)
		
		# Fallback to raw parsing
		raw_content = str(raw_msg.content)
		cleaned_content = clean_response_content(raw_content)
		parsed_json = extract_json_from_model_output(cleaned_content)
		return output_schema(**parsed_json)
	
	def _extract_raw_content(self, response: Any) -> str:
		"""Extract raw content from various response formats."""
		if isinstance(response, dict) and 'raw' in response:
			return str(response['raw'].content)
		elif hasattr(response, 'content'):
			return str(response.content)
		else:
			return str(response)
	
	def _add_schema_as_system_message(
		self, 
		messages: list[BaseMessage], 
		output_schema: type[OutputType]
	) -> list[BaseMessage]:
		"""Add schema format to first system message or create one if none exists."""
		from langchain_core.messages import SystemMessage
		
		schema_description = self._generate_full_schema_description(output_schema)
		
		if not schema_description:
			return messages
			
		modified_messages = messages.copy()
		
		# Find first system message
		first_system_index = None
		for i, msg in enumerate(modified_messages):
			if isinstance(msg, SystemMessage):
				first_system_index = i
				break
		
		if first_system_index is not None:
			# Append schema to existing first system message
			existing_content = modified_messages[first_system_index].content
			new_content = f"{existing_content}\n\n{schema_description}"
			modified_messages[first_system_index] = SystemMessage(content=new_content)
		else:
			# No system message exists, add one at the beginning
			schema_system_message = SystemMessage(content=schema_description)
			modified_messages.insert(0, schema_system_message)
		
		return modified_messages
	
	def _generate_full_schema_description(self, output_schema: type[OutputType]) -> str:
		"""Generate complete schema description for raw method models without $refs."""
		try:
			# Get schema with all references resolved
			schema = output_schema.model_json_schema(mode='serialization')
			resolved_schema = self._resolve_schema_refs(schema)
			
			# Generate compact schema format
			compact_schema = self._generate_compact_schema(resolved_schema)
			
			# Generate system prompt with compact schema
			format_description = "OUTPUT FORMAT: You must respond with valid JSON matching this exact schema:\n\n"
			format_description += compact_schema
			
			# Add explanation
			format_description += "\n\nRules:\n"
			format_description += "- Your response must be valid JSON only\n"
			format_description += "- Follow the schema structure exactly\n"
			format_description += "- All required fields must be provided\n"
			
			return format_description
		except Exception as e:
			logger.warning(f"Failed to generate full schema description: {e}")
			return ""
	
	def _resolve_schema_refs(self, schema: dict) -> dict:
		"""Resolve all $ref references in a JSON schema to create a fully expanded schema."""
		if not isinstance(schema, dict):
			return schema
			
		definitions = schema.get('$defs', {})
		
		def resolve_refs(obj):
			if isinstance(obj, dict):
				if '$ref' in obj:
					# Extract reference path
					ref_path = obj['$ref']
					if ref_path.startswith('#/$defs/'):
						ref_name = ref_path[8:]  # Remove '#/$defs/'
						if ref_name in definitions:
							# Recursively resolve the referenced schema
							return resolve_refs(definitions[ref_name])
					return obj
				else:
					# Recursively process all dict values
					return {k: resolve_refs(v) for k, v in obj.items() if k != '$defs'}
			elif isinstance(obj, list):
				return [resolve_refs(item) for item in obj]
			else:
				return obj
		
		return resolve_refs(schema)
	
	def _generate_compact_schema(self, schema: dict) -> str:
		"""Generate a compact schema format like Python type hints."""
		def format_type(obj: dict | str | list, is_array_item: bool = False) -> str:
			if isinstance(obj, str):
				return obj
			
			if isinstance(obj, list):
				# List elements separated by comma, no | null for array items
				formatted_items = []
				for item in obj:
					formatted_items.append(format_type(item, True))
				return f"[{', '.join(formatted_items)}]"
			
			if not isinstance(obj, dict):
				return str(obj)
			
			# Handle anyOf for optional types
			if 'anyOf' in obj:
				types = []
				for item in obj['anyOf']:
					if item.get('type') == 'null':
						continue
					types.append(format_type(item, is_array_item))
				
				# For array items, never add | null - just return the type
				if is_array_item:
					return types[0] if len(types) == 1 else f"({' | '.join(types)})"
				
				# For non-array items, check if null is in anyOf (making it optional)
				has_null = any(item.get('type') == 'null' for item in obj['anyOf'])
				type_str = ' | '.join(types)
				if has_null and len(types) == 1:
					type_str += ' | null'
				return type_str
			
			# Handle basic types
			if 'type' in obj:
				base_type = obj['type']
				
				if base_type == 'string':
					return 'str'
				elif base_type == 'integer':
					return 'int'
				elif base_type == 'number':
					return 'float'
				elif base_type == 'boolean':
					return 'bool'
				elif base_type == 'array':
					if 'items' in obj:
						item_type = format_type(obj['items'], True)
						return f"[{item_type}]"
					return '[]'
				elif base_type == 'object':
					return format_object(obj, is_nested=True, is_array_item=is_array_item)
			
			return format_object(obj, is_nested=True)
		
		def format_object(obj: dict, is_nested: bool = False, is_array_item: bool = False) -> str:
			if 'properties' not in obj:
				return 'object'
			
			properties = obj['properties']
			required = set(obj.get('required', []))
			
			lines = []

			desc = ""
			if not is_nested and 'description' in obj:
				desc = f"# {obj['description']} "
			
			for field_name, field_schema in properties.items():
				# Get the field description first
				field_desc = ""
				if 'description' in field_schema:
					field_desc = f"  # {field_schema['description']}\n"
				
				field_type = format_type(field_schema)
				
				# Check for defaults
				default_val = field_schema.get('default')
				
				# Check if field is required
				is_required = field_name in required
				
				# Build the field line
				if not is_required and default_val is not None:
					# Optional with default
					if isinstance(default_val, str):
						field_line = f"  {field_name}: {field_type} = '{default_val}'"
					else:
						field_line = f"  {field_name}: {field_type} = {default_val}"
				elif is_required or is_array_item:
					# Required field
					field_line = f"  {field_name}: {field_type}"
				else:
					# Optional without default (add null to type only if not array item)
					if ' | null' not in field_type and not is_array_item:
						field_type += ' | null'
					field_line = f"  {field_name}: {field_type}"
				
				# Add the field with description above it if present
				if field_desc:
					lines.append(field_desc.rstrip())  # Remove trailing newline
					lines.append(field_line)
				else:
					lines.append(field_line)
			
			# Always return just braces, no object name
			if lines:
				return desc + "{\n" + '\n'.join(lines) + "\n}"
			else:
				return desc + "{}"
		
		return format_object(schema)
	
	
	
