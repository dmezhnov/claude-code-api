"""Chat completions API endpoint - OpenAI compatible."""

import uuid
import json
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import ValidationError
import structlog

from claude_code_api.models.openai import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    ChatCompletionUsage,
    ErrorResponse
)
from claude_code_api.models.claude import validate_claude_model, get_model_info
from claude_code_api.core.claude_manager import create_project_directory
from claude_code_api.core.session_manager import SessionManager, ConversationManager
from claude_code_api.utils.streaming import create_sse_response, create_non_streaming_response
from claude_code_api.utils.parser import ClaudeOutputParser, estimate_tokens
from claude_code_api.utils.tools import format_tools_prompt, parse_tool_calls

logger = structlog.get_logger()
router = APIRouter()


async def _wrap_as_sse(response: dict):
    """Wrap a non-streaming response dict as SSE events.

    Converts a chat.completion object into streaming chunks so that
    clients expecting SSE (text/event-stream) can consume it.
    """
    import uuid as _uuid
    from datetime import datetime as _dt

    completion_id = response.get("id", f"chatcmpl-{_uuid.uuid4().hex[:29]}")
    created = response.get("created", int(_dt.utcnow().timestamp()))
    model = response.get("model", "unknown")

    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    finish_reason = choice.get("finish_reason", "stop")

    # Initial chunk with role
    initial = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(initial)}\n\n"

    # Content chunk
    if content:
        content_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

    # Tool call chunks (OpenAI streaming tool_calls format)
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            tc_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }]},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(tc_chunk)}\n\n"

    # Final chunk
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def create_chat_completion(
    req: Request
) -> Any:
    """Create a chat completion, compatible with OpenAI API."""
    
    # Log raw request for debugging
    try:
        raw_body = await req.body()
        content_type = req.headers.get("content-type", "unknown")
        logger.info(
            "Raw request received",
            content_type=content_type,
            body_size=len(raw_body),
            user_agent=req.headers.get("user-agent", "unknown"),
            raw_body=raw_body.decode()[:1000] if raw_body else "empty"
        )
        
        # Parse JSON manually to see validation errors
        if raw_body:
            try:
                json_data = json.loads(raw_body.decode())
                logger.info("JSON parsed successfully", data_keys=list(json_data.keys()))
            except json.JSONDecodeError as e:
                logger.error("JSON decode error", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": {"message": f"Invalid JSON: {str(e)}", "type": "invalid_request_error"}}
                )
        
        # Try to validate with Pydantic
        try:
            request = ChatCompletionRequest(**json_data)
            logger.info("Pydantic validation successful")
        except ValidationError as e:
            logger.error("Pydantic validation failed", error=str(e), errors=e.errors())
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": {"message": f"Validation error: {str(e)}", "type": "invalid_request_error", "details": e.errors()}}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process request", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Internal server error", "type": "internal_error"}}
        )
    
    # Get managers from app state
    session_manager: SessionManager = req.app.state.session_manager
    claude_manager = req.app.state.claude_manager

    # Extract client info for logging
    client_id = getattr(req.state, 'client_id', 'anonymous')

    # When tools are present, we collect the full response (non-streaming)
    # for tool_call parsing, but may still wrap the result as SSE if the
    # client requested streaming.
    has_tools = bool(request.tools)
    wants_stream = request.stream
    if has_tools and request.stream:
        logger.info("Collecting full response for tool_call parsing (will wrap as SSE)")
        request.stream = False

    logger.info(
        "Chat completion request validated",
        client_id=client_id,
        model=request.model,
        messages_count=len(request.messages),
        stream=request.stream,
        project_id=request.project_id,
        session_id=request.session_id
    )
    
    try:
        # Validate model
        claude_model = validate_claude_model(request.model)
        model_info = get_model_info(claude_model)
        
        # Validate message format
        if not request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "At least one message is required",
                        "type": "invalid_request_error",
                        "code": "missing_messages"
                    }
                }
            )
        
        # Build conversation prompt from all messages.
        # Keep the first system message as the system prompt (extracted later)
        # but include subsequent system messages (e.g. cron events) in history.
        system_messages_all = [msg for msg in request.messages if msg.role == "system"]
        conversation_messages = []
        first_system_seen = False
        for msg in request.messages:
            if msg.role == "system":
                if first_system_seen:
                    # Subsequent system messages (cron events etc.) go into history
                    conversation_messages.append(msg)
                else:
                    first_system_seen = True
                    # Skip the first system message (main system prompt)
            else:
                conversation_messages.append(msg)
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "At least one user message is required",
                        "type": "invalid_request_error",
                        "code": "missing_user_message"
                    }
                }
            )

        last_user_msg = user_messages[-1]

        # If there are previous messages, format as conversation history
        if len(conversation_messages) > 1:
            parts = []
            for msg in conversation_messages:
                if msg.role == "user":
                    parts.append(f"[User]: {msg.get_text_content()}")
                elif msg.role == "assistant":
                    text = msg.get_text_content() or ""
                    # Include tool calls made by the assistant
                    if msg.tool_calls:
                        import json as _json
                        for tc in msg.tool_calls:
                            text += f"\n[Called tool: {tc.function.name}({tc.function.arguments})]"
                    parts.append(f"[Assistant]: {text}")
                elif msg.role == "system":
                    # Subsequent system messages (cron events, notifications)
                    parts.append(f"[System Event]: {msg.get_text_content()}")
                elif msg.role == "tool":
                    # Tool result from gateway
                    tool_name = msg.name or "unknown"
                    parts.append(f"[Tool Result ({tool_name})]: {msg.get_text_content()}")
            user_prompt = (
                "Below is the conversation history. Continue naturally from where it left off. "
                "Reply ONLY as the Assistant to the last User message.\n\n"
                + "\n\n".join(parts)
            )
        else:
            user_prompt = last_user_msg.get_text_content()

        # Handle vision: extract images and prepend Read instructions
        image_paths = last_user_msg.extract_images()
        if image_paths:
            image_refs = "\n".join(
                f"- Image {i+1}: {path}" for i, path in enumerate(image_paths)
            )
            user_prompt = (
                f"Read the following image file(s) using the Read tool, "
                f"then answer the question below.\n\n"
                f"Image files:\n{image_refs}\n\n"
                f"Question: {user_prompt}"
            )
            logger.info(
                "Vision request: extracted images",
                image_count=len(image_paths),
                image_paths=image_paths
            )

        # Extract system prompt
        system_messages = [msg for msg in request.messages if msg.role == "system"]
        system_prompt = system_messages[0].get_text_content() if system_messages else request.system_prompt

        # Build tool prompt if tools are provided
        append_system_prompt = None
        if has_tools:
            append_system_prompt = format_tools_prompt(request.tools)
            logger.info("Tools provided", tool_count=len(request.tools))

        # Handle project context
        project_id = request.project_id or f"default-{client_id}"
        project_path = create_project_directory(project_id)
        
        # Handle session management
        if request.session_id:
            # Continue existing session
            session_id = request.session_id
            session_info = await session_manager.get_session(session_id)
            
            if not session_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": {
                            "message": f"Session {session_id} not found",
                            "type": "invalid_request_error",
                            "code": "session_not_found"
                        }
                    }
                )
        else:
            # Create new session
            session_id = await session_manager.create_session(
                project_id=project_id,
                model=claude_model,
                system_prompt=system_prompt
            )
        
        # Start Claude Code process
        try:
            claude_process = await claude_manager.create_session(
                session_id=session_id,
                project_path=project_path,
                prompt=user_prompt,
                model=claude_model,
                system_prompt=system_prompt,
                resume_session=request.session_id,
                append_system_prompt=append_system_prompt,
                disable_builtin_tools=has_tools,
            )
        except Exception as e:
            logger.error(
                "Failed to create Claude session",
                session_id=session_id,
                error=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": f"Failed to start Claude Code: {str(e)}",
                        "type": "service_unavailable",
                        "code": "claude_unavailable"
                    }
                }
            )
        
        # Use Claude's actual session ID
        claude_session_id = claude_process.session_id
        
        # Update session with user message
        await session_manager.update_session(
            session_id=claude_session_id,
            message_content=user_prompt,
            role="user",
            tokens_used=estimate_tokens(user_prompt)
        )
        
        # Handle streaming vs non-streaming
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                create_sse_response(claude_session_id, claude_model, claude_process),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Session-ID": claude_session_id,
                    "X-Project-ID": project_id
                }
            )
        else:
            # Collect all output for non-streaming response
            messages = []
            
            async for claude_message in claude_process.get_output():
                # Log each message from Claude
                logger.info(
                    "Received Claude message",
                    message_type=claude_message.get("type") if isinstance(claude_message, dict) else type(claude_message).__name__,
                    message_keys=list(claude_message.keys()) if isinstance(claude_message, dict) else [],
                    has_assistant_content=bool(isinstance(claude_message, dict) and 
                                             claude_message.get("type") == "assistant" and 
                                             claude_message.get("message", {}).get("content")),
                    message_preview=str(claude_message)[:200] if claude_message else "None"
                )
                
                messages.append(claude_message)
                
                # Check if it's a final message by looking at dict structure
                is_final = False
                if isinstance(claude_message, dict):
                    is_final = claude_message.get("type") == "result"
                
                # Stop on final message or after a reasonable number of messages
                if is_final or len(messages) > 10:  # Safety limit for testing
                    break
            
            # Log what we collected
            logger.info(
                "Claude messages collected", 
                total_messages=len(messages),
                message_types=[msg.get("type") if isinstance(msg, dict) else type(msg).__name__ for msg in messages]
            )
            
            # Simple usage tracking without parsing Claude internals
            usage_summary = {"total_tokens": 50, "total_cost": 0.001}
            await session_manager.update_session(
                session_id=claude_session_id,
                tokens_used=50,
                cost=0.001
            )
            
            # Create non-streaming response
            response = create_non_streaming_response(
                messages=messages,
                session_id=claude_session_id,
                model=claude_model,
                usage_summary=usage_summary
            )

            # Parse tool calls from response text if tools were provided
            if has_tools and response.get("choices"):
                choice = response["choices"][0]
                content = choice.get("message", {}).get("content", "")
                if content:
                    tool_calls, cleaned_text = parse_tool_calls(content)
                    if tool_calls:
                        # Drop text content when tool_calls are present.
                        # OpenClaw will show content as a message AND execute
                        # tools, then send results back for another completion.
                        # This creates duplicate messages.  Let the final
                        # completion (after tool execution) provide the
                        # user-facing text instead.
                        choice["message"]["content"] = None
                        choice["message"]["tool_calls"] = [
                            tc.model_dump() for tc in tool_calls
                        ]
                        choice["finish_reason"] = "tool_calls"
                        logger.info(
                            "Tool calls parsed from response",
                            tool_count=len(tool_calls),
                            tools=[tc.function.name for tc in tool_calls],
                        )

            # Add extension fields
            response["project_id"] = project_id

            # If the client originally requested streaming, wrap as SSE
            if wants_stream:
                return StreamingResponse(
                    _wrap_as_sse(response),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Session-ID": claude_session_id,
                        "X-Project-ID": project_id,
                    },
                )

            return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            "Unexpected error in chat completion",
            client_id=client_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                    "code": "unexpected_error"
                }
            }
        )


@router.get("/chat/completions/{session_id}/status")
async def get_completion_status(
    session_id: str,
    req: Request
) -> Dict[str, Any]:
    """Get status of a chat completion session."""
    
    session_manager: SessionManager = req.app.state.session_manager
    claude_manager = req.app.state.claude_manager
    
    # Get session info
    session_info = await session_manager.get_session(session_id)
    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"Session {session_id} not found",
                    "type": "not_found",
                    "code": "session_not_found"
                }
            }
        )
    
    # Get Claude process status
    claude_process = await claude_manager.get_session(session_id)
    is_running = claude_process is not None and claude_process.is_running
    
    return {
        "session_id": session_id,
        "project_id": session_info.project_id,
        "model": session_info.model,
        "is_running": is_running,
        "created_at": session_info.created_at.isoformat(),
        "updated_at": session_info.updated_at.isoformat(),
        "total_tokens": session_info.total_tokens,
        "total_cost": session_info.total_cost,
        "message_count": session_info.message_count
    }


@router.post("/chat/completions/debug")
async def debug_chat_completion(req: Request) -> Dict[str, Any]:
    """Debug endpoint to test request validation."""
    try:
        raw_body = await req.body()
        headers = dict(req.headers)
        
        logger.info(
            "Debug request",
            content_type=headers.get("content-type"),
            body_size=len(raw_body),
            headers=headers,
            raw_body=raw_body.decode() if raw_body else "empty"
        )
        
        if raw_body:
            json_data = json.loads(raw_body.decode())
            
            # Try validation
            try:
                request = ChatCompletionRequest(**json_data)
                return {
                    "status": "success",
                    "message": "Request validation passed",
                    "parsed_data": {
                        "model": request.model,
                        "messages_count": len(request.messages),
                        "stream": request.stream
                    }
                }
            except ValidationError as e:
                return {
                    "status": "validation_error",
                    "message": str(e),
                    "errors": e.errors(),
                    "raw_data": json_data
                }
        
        return {"status": "no_body"}
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@router.delete("/chat/completions/{session_id}")
async def stop_completion(
    session_id: str,
    req: Request
) -> Dict[str, str]:
    """Stop a running chat completion session."""
    
    session_manager: SessionManager = req.app.state.session_manager
    claude_manager = req.app.state.claude_manager
    
    # Stop Claude process
    await claude_manager.stop_session(session_id)
    
    # End session
    await session_manager.end_session(session_id)
    
    logger.info("Chat completion stopped", session_id=session_id)
    
    return {
        "session_id": session_id,
        "status": "stopped"
    }
