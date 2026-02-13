"""Utilities for OpenAI-compatible tool calling via Claude CLI."""

import json
import re
import uuid
from typing import List, Optional, Tuple

from claude_code_api.models.openai import Tool, ToolCall, FunctionCall


def generate_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:24]}"


def format_tools_prompt(tools: List[Tool]) -> str:
    """Convert OpenAI tool definitions into a system prompt appendix.

    Instructs Claude to output tool calls in a fenced code block
    format that can be reliably parsed from the response text.
    """
    if not tools:
        return ""

    tool_descriptions = []
    for tool in tools:
        fn = tool.function
        desc = f"- **{fn.name}**"
        if fn.description:
            desc += f": {fn.description}"
        if fn.parameters:
            # Include only a compact schema summary
            params = fn.parameters.get("properties", {})
            required = fn.parameters.get("required", [])
            if params:
                param_parts = []
                for pname, pinfo in params.items():
                    ptype = pinfo.get("type", "any")
                    req = " (required)" if pname in required else ""
                    pdesc = pinfo.get("description", "")
                    param_parts.append(f"  - `{pname}` ({ptype}{req}): {pdesc}")
                desc += "\n" + "\n".join(param_parts)
        tool_descriptions.append(desc)

    return (
        "# Available Tools\n\n"
        "You have access to the following tools. To call a tool, output EXACTLY "
        "this format (use a fenced code block with the language tag `tool_call`):\n\n"
        "```tool_call\n"
        '{"name": "tool_name", "arguments": {"param": "value"}}\n'
        "```\n\n"
        "Rules:\n"
        "- You may include text before and/or after tool calls.\n"
        "- You may call multiple tools in one response (use separate blocks).\n"
        "- The arguments value must be a JSON object matching the tool's parameters.\n"
        "- ALWAYS use this exact format when you want to perform an action.\n\n"
        "Tools:\n\n" + "\n\n".join(tool_descriptions)
    )


TOOL_CALL_PATTERN = re.compile(
    r"```tool_call\s*\n(.*?)\n```",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> Tuple[Optional[List[ToolCall]], str]:
    """Parse tool_call fenced blocks from Claude's response text.

    Returns:
        Tuple of (list of ToolCall objects or None, cleaned text with blocks removed).
    """
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        return None, text

    tool_calls = []
    for match in matches:
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        name = data.get("name")
        arguments = data.get("arguments", {})
        if not name:
            continue

        tool_calls.append(
            ToolCall(
                id=generate_tool_call_id(),
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                ),
            )
        )

    if not tool_calls:
        return None, text

    # Remove tool_call blocks from text
    cleaned = TOOL_CALL_PATTERN.sub("", text).strip()

    return tool_calls, cleaned
