"""Tool format conversion between OpenAI and Anthropic formats."""

import json
import uuid
from typing import List, Optional, Tuple, Any

from claude_code_api.models.openai import Tool, ToolCall, FunctionCall


def generate_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:24]}"


def openai_tools_to_anthropic(tools: List[Tool]) -> List[dict]:
    """Convert OpenAI tool definitions to Anthropic format.

    OpenAI:  {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
    Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
    """
    result = []
    for tool in tools:
        fn = tool.function
        result.append({
            "name": fn.name,
            "description": fn.description or "",
            "input_schema": fn.parameters or {"type": "object", "properties": {}},
        })
    return result


def anthropic_tool_use_to_openai(
    content_blocks: List[Any],
) -> Tuple[Optional[List[ToolCall]], Optional[str]]:
    """Convert Anthropic content blocks to OpenAI ToolCall objects + text.

    Anthropic response content:
        [{"type": "text", "text": "..."}, {"type": "tool_use", "id": "...", "name": "...", "input": {...}}]

    Returns (tool_calls or None, text_content or None).
    """
    tool_calls: List[ToolCall] = []
    text_parts: List[str] = []

    for block in content_blocks:
        # Handle SDK objects (have .type attribute)
        if hasattr(block, "type"):
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    type="function",
                    function=FunctionCall(
                        name=block.name,
                        arguments=json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                    ),
                ))
        # Handle dict format (from serialized data)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", generate_tool_call_id()),
                    type="function",
                    function=FunctionCall(
                        name=block["name"],
                        arguments=json.dumps(block.get("input", {})),
                    ),
                ))

    text = "\n".join(text_parts).strip() if text_parts else None
    return (tool_calls if tool_calls else None, text)
