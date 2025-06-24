from __future__ import annotations

import json
import logging
import os
from typing import Any

import anyio
from langchain_core.messages import (
	BaseMessage,
)

logger = logging.getLogger(__name__)


async def save_conversation(
	input_messages: list[BaseMessage], response: Any, target: str | Path, encoding: str | None = None
) -> None:
	"""Save conversation history to file asynchronously."""
	target_path = Path(target)

	# create folders if not exists
	if target_path.parent:
		await anyio.Path(target_path.parent).mkdir(parents=True, exist_ok=True)

	await anyio.Path(target_path).write_text(await _format_conversation(input_messages, response), encoding=encoding or 'utf-8')


async def _format_conversation(messages: list[BaseMessage], response: Any) -> str:
	"""Format the conversation including messages and response."""
	lines = []

	# Format messages
	for message in messages:
		lines.append(f' {message.__class__.__name__} ')

		if isinstance(message.content, list):
			for item in message.content:
				if isinstance(item, dict) and item.get('type') == 'text':
					lines.append(item['text'].strip())
		elif isinstance(message.content, str):
			try:
				content = json.loads(message.content)
				lines.append(json.dumps(content, indent=2))
			except json.JSONDecodeError:
				lines.append(message.content.strip())

		lines.append('')  # Empty line after each message

	# Format response
	lines.append(' RESPONSE')
	lines.append(json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2))

	return '\n'.join(lines)


# Note: _write_messages_to_file and _write_response_to_file have been merged into _format_conversation
# This is more efficient for async operations and reduces file I/O
