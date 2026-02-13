"""Chat completions API endpoint - OpenAI compatible.

Converts between OpenAI chat format and Anthropic Messages API,
calling the Anthropic SDK directly instead of spawning a CLI subprocess.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator

import anthropic
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import ValidationError
import structlog

from claude_code_api.models.openai import (
    ChatCompletionRequest,
    ChatMessage,
    ToolCall,
)
from claude_code_api.models.claude import validate_claude_model, get_model_info
from claude_code_api.core.session_manager import SessionManager
from claude_code_api.utils.tools import openai_tools_to_anthropic, anthropic_tool_use_to_openai

logger = structlog.get_logger()
router = APIRouter()


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic message conversion
# ---------------------------------------------------------------------------

def convert_messages(
    messages: List[ChatMessage],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert OpenAI messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    """
    system_prompt: Optional[str] = None
    anthropic_msgs: List[Dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            if system_prompt is None:
                system_prompt = msg.get_text_content()
            else:
                # Subsequent system messages injected as user text
                anthropic_msgs.append({
                    "role": "user",
                    "content": f"[System notification]: {msg.get_text_content()}",
                })

        elif msg.role == "user":
            content = _build_user_content(msg)
            anthropic_msgs.append({"role": "user", "content": content})

        elif msg.role == "assistant":
            content = _build_assistant_content(msg)
            anthropic_msgs.append({"role": "assistant", "content": content})

        elif msg.role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id or "unknown",
                "content": msg.get_text_content(),
            }
            # Anthropic requires tool_result inside a user message
            if anthropic_msgs and anthropic_msgs[-1]["role"] == "user":
                prev = anthropic_msgs[-1]["content"]
                if isinstance(prev, list):
                    prev.append(tool_result_block)
                else:
                    anthropic_msgs[-1]["content"] = [
                        {"type": "text", "text": prev} if isinstance(prev, str) else prev,
                        tool_result_block,
                    ]
            else:
                anthropic_msgs.append({
                    "role": "user",
                    "content": [tool_result_block],
                })

    anthropic_msgs = _fix_alternation(anthropic_msgs)
    return system_prompt, anthropic_msgs


def _build_user_content(msg: ChatMessage) -> Any:
    """Build user content, handling text and images."""
    if not isinstance(msg.content, list):
        return msg.get_text_content()

    blocks: List[Dict[str, Any]] = []
    for item in msg.content:
        if not isinstance(item, dict):
            blocks.append({"type": "text", "text": str(item)})
            continue
        if item.get("type") == "text":
            blocks.append({"type": "text", "text": item.get("text", "")})
        elif item.get("type") == "image_url":
            image_data = item.get("image_url", "")
            url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data)
            if url.startswith("data:"):
                # Parse data URI: data:image/png;base64,<data>
                header, _, b64 = url.partition(";base64,")
                media_type = header.replace("data:", "")
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    },
                })
    return blocks if blocks else msg.get_text_content()


def _build_assistant_content(msg: ChatMessage) -> Any:
    """Build assistant content, including tool_use blocks."""
    blocks: List[Dict[str, Any]] = []
    text = msg.get_text_content()
    if text:
        blocks.append({"type": "text", "text": text})

    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.function.name,
                "input": args,
            })

    if blocks:
        return blocks
    return text or ""


def _fix_alternation(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure strict user/assistant alternation required by Anthropic API."""
    if not messages:
        return messages

    fixed = [messages[0]]
    for msg in messages[1:]:
        if msg["role"] == fixed[-1]["role"]:
            _merge_content(fixed[-1], msg)
        else:
            fixed.append(msg)

    # First message must be user
    if fixed and fixed[0]["role"] != "user":
        fixed.insert(0, {"role": "user", "content": "Continue."})

    return fixed


def _merge_content(target: Dict[str, Any], source: Dict[str, Any]):
    """Merge source content into target (same role)."""
    tc = target["content"]
    sc = source["content"]

    def _to_list(c: Any) -> list:
        if isinstance(c, list):
            return c
        if isinstance(c, str):
            return [{"type": "text", "text": c}]
        return [c]

    target["content"] = _to_list(tc) + _to_list(sc)


# ---------------------------------------------------------------------------
# Anthropic response -> OpenAI response conversion
# ---------------------------------------------------------------------------

def _stop_reason_to_finish(stop_reason: Optional[str]) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    if stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    return "stop"


def _build_openai_response(
    response: anthropic.types.Message,
    model: str,
    session_id: str,
) -> Dict[str, Any]:
    """Convert Anthropic Message to OpenAI ChatCompletion dict."""
    tool_calls_out, text_content = anthropic_tool_use_to_openai(response.content)
    finish_reason = _stop_reason_to_finish(response.stop_reason)

    message: Dict[str, Any] = {"role": "assistant", "content": text_content}
    if tool_calls_out:
        message["tool_calls"] = [tc.model_dump() for tc in tool_calls_out]
        # When tool_calls present, drop text to avoid duplicate display
        message["content"] = None

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(datetime.utcnow().timestamp()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
        "session_id": session_id,
    }


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------

async def _wrap_as_sse(response: dict) -> AsyncGenerator[str, None]:
    """Wrap a non-streaming response dict as SSE events."""
    completion_id = response.get("id", f"chatcmpl-{uuid.uuid4().hex[:29]}")
    created = response.get("created", int(datetime.utcnow().timestamp()))
    model = response.get("model", "unknown")

    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    finish_reason = choice.get("finish_reason", "stop")

    # Role chunk
    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    if content:
        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': i, 'id': tc['id'], 'type': 'function', 'function': {'name': tc['function']['name'], 'arguments': tc['function']['arguments']}}]}, 'finish_reason': None}]})}\n\n"

    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_anthropic_as_sse(
    session,
    anthropic_messages: List[Dict],
    model: str,
    system_prompt: Optional[str],
    anthropic_tools: Optional[List[Dict]],
    max_tokens: int,
    temperature: Optional[float],
    session_id: str,
) -> AsyncGenerator[str, None]:
    """Stream Anthropic response as OpenAI-compatible SSE chunks."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(datetime.utcnow().timestamp())

    # Role chunk
    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    stream_cm = await session.stream_message(
        messages=anthropic_messages,
        model=model,
        system_prompt=system_prompt,
        tools=anthropic_tools,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    finish_reason = "stop"

    async with stream_cm as stream:
        async for text in stream.text_stream:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        final = await stream.get_final_message()

        if final.stop_reason == "tool_use":
            finish_reason = "tool_calls"
            tool_calls_out, _ = anthropic_tool_use_to_openai(final.content)
            if tool_calls_out:
                for i, tc in enumerate(tool_calls_out):
                    tc_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [{
                            "index": i,
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }]}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(tc_chunk)}\n\n"
        elif final.stop_reason == "max_tokens":
            finish_reason = "length"

    # Final chunk
    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
async def create_chat_completion(req: Request) -> Any:
    """Create a chat completion, compatible with OpenAI API."""

    # Parse request body
    try:
        raw_body = await req.body()
        logger.info(
            "Raw request received",
            body_size=len(raw_body),
            user_agent=req.headers.get("user-agent", "unknown"),
        )

        if not raw_body:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"message": "Empty request body", "type": "invalid_request_error"}},
            )

        try:
            json_data = json.loads(raw_body.decode())
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}},
            )

        try:
            request = ChatCompletionRequest(**json_data)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": {"message": f"Validation error: {e}", "type": "invalid_request_error"}},
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process request", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Internal server error", "type": "internal_error"}},
        )

    # Managers
    session_manager: SessionManager = req.app.state.session_manager
    claude_manager = req.app.state.claude_manager
    client_id = getattr(req.state, "client_id", "anonymous")

    # For tool calls we collect the full response then optionally wrap as SSE
    has_tools = bool(request.tools)
    wants_stream = request.stream
    if has_tools and request.stream:
        logger.info("Collecting full response for tool_call parsing (will wrap as SSE)")
        request.stream = False

    logger.info(
        "Chat completion request",
        model=request.model,
        messages_count=len(request.messages),
        stream=request.stream,
        has_tools=has_tools,
    )

    try:
        # Validate model
        claude_model = validate_claude_model(request.model)

        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": {"message": "At least one message is required", "type": "invalid_request_error"}},
            )

        # Convert messages
        system_prompt, anthropic_messages = convert_messages(request.messages)

        # Override system prompt if provided via extension field
        if request.system_prompt and not system_prompt:
            system_prompt = request.system_prompt

        # Convert tools
        anthropic_tools = None
        if has_tools:
            anthropic_tools = openai_tools_to_anthropic(request.tools)
            logger.info("Tools converted", tool_count=len(request.tools))

        # Max tokens
        max_tokens = request.max_tokens or 16384

        # Session management
        project_id = request.project_id or f"default-{client_id}"
        if request.session_id:
            session_id = request.session_id
            session_info = await session_manager.get_session(session_id)
            if not session_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": {"message": f"Session {session_id} not found", "type": "invalid_request_error"}},
                )
        else:
            session_id = await session_manager.create_session(
                project_id=project_id,
                model=claude_model,
                system_prompt=system_prompt,
            )

        # Create Anthropic session
        try:
            session = await claude_manager.create_session(session_id)
        except Exception as e:
            logger.error("Failed to create session", session_id=session_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": {"message": f"Service unavailable: {e}", "type": "service_unavailable"}},
            )

        # Streaming response
        if request.stream:
            return StreamingResponse(
                _stream_anthropic_as_sse(
                    session=session,
                    anthropic_messages=anthropic_messages,
                    model=claude_model,
                    system_prompt=system_prompt,
                    anthropic_tools=anthropic_tools,
                    max_tokens=max_tokens,
                    temperature=request.temperature,
                    session_id=session_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Session-ID": session_id,
                },
            )

        # Non-streaming response
        try:
            response_msg = await session.create_message(
                messages=anthropic_messages,
                model=claude_model,
                system_prompt=system_prompt,
                tools=anthropic_tools,
                max_tokens=max_tokens,
                temperature=request.temperature,
            )
        except anthropic.APIError as e:
            logger.error("Anthropic API error", error=str(e), status_code=getattr(e, "status_code", None))
            status_code = getattr(e, "status_code", 500)
            if status_code == 429:
                raise HTTPException(status_code=429, detail={"error": {"message": "Rate limited", "type": "rate_limit_error"}})
            raise HTTPException(
                status_code=status_code or 500,
                detail={"error": {"message": str(e), "type": "api_error"}},
            )

        # Update session tracking
        await session_manager.update_session(
            session_id=session_id,
            tokens_used=response_msg.usage.input_tokens + response_msg.usage.output_tokens,
            cost=0.0,
        )

        # Build OpenAI response
        response = _build_openai_response(response_msg, claude_model, session_id)
        response["project_id"] = project_id

        # Clean up session
        await claude_manager.stop_session(session_id)

        # If client wanted streaming, wrap the collected response as SSE
        if wants_stream:
            return StreamingResponse(
                _wrap_as_sse(response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Session-ID": session_id,
                },
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in chat completion", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Internal server error", "type": "internal_error"}},
        )


@router.get("/chat/completions/{session_id}/status")
async def get_completion_status(session_id: str, req: Request) -> Dict[str, Any]:
    """Get status of a chat completion session."""
    session_manager: SessionManager = req.app.state.session_manager

    session_info = await session_manager.get_session(session_id)
    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": {"message": f"Session {session_id} not found", "type": "not_found"}},
        )

    return {
        "session_id": session_id,
        "project_id": session_info.project_id,
        "model": session_info.model,
        "is_running": False,
        "created_at": session_info.created_at.isoformat(),
        "updated_at": session_info.updated_at.isoformat(),
        "total_tokens": session_info.total_tokens,
        "total_cost": session_info.total_cost,
        "message_count": session_info.message_count,
    }


@router.delete("/chat/completions/{session_id}")
async def stop_completion(session_id: str, req: Request) -> Dict[str, str]:
    """Stop a running chat completion session."""
    session_manager: SessionManager = req.app.state.session_manager
    claude_manager = req.app.state.claude_manager

    await claude_manager.stop_session(session_id)
    await session_manager.end_session(session_id)

    logger.info("Chat completion stopped", session_id=session_id)
    return {"session_id": session_id, "status": "stopped"}
