"""Anthropic API client management.

Provides AnthropicSession (single API call wrapper) and ClaudeManager
(session lifecycle manager) that replace the old subprocess-based approach.
"""

import os
from typing import Optional, Dict, List, Any, AsyncGenerator

import anthropic
import structlog

from .config import settings

logger = structlog.get_logger()


class AnthropicSession:
    """Wraps a single Anthropic API interaction."""

    def __init__(self, session_id: str, client: anthropic.AsyncAnthropic):
        self.session_id = session_id
        self.client = client

    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 16384,
        temperature: Optional[float] = None,
    ) -> anthropic.types.Message:
        """Non-streaming Anthropic API call."""
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature

        logger.info(
            "Calling Anthropic API",
            session_id=self.session_id,
            model=model,
            message_count=len(messages),
            max_tokens=max_tokens,
            has_tools=bool(tools),
        )

        return await self.client.messages.create(**kwargs)

    async def stream_message(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 16384,
        temperature: Optional[float] = None,
    ) -> anthropic.MessageStream:
        """Return a streaming context manager for the Anthropic API.

        Usage:
            stream_cm = await session.stream_message(...)
            async with stream_cm as stream:
                async for text in stream.text_stream:
                    ...
        """
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature

        logger.info(
            "Starting Anthropic stream",
            session_id=self.session_id,
            model=model,
            message_count=len(messages),
            max_tokens=max_tokens,
            has_tools=bool(tools),
        )

        return self.client.messages.stream(**kwargs)


class ClaudeManager:
    """Manages Anthropic API sessions."""

    def __init__(self):
        api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key if api_key else anthropic.MISSING,
        )
        self.active_sessions: Dict[str, AnthropicSession] = {}
        self.max_concurrent = settings.max_concurrent_sessions

    async def get_version(self) -> str:
        """Return Anthropic SDK version."""
        return f"anthropic-sdk-{anthropic.__version__}"

    async def create_session(self, session_id: str) -> AnthropicSession:
        """Create a new API session."""
        if len(self.active_sessions) >= self.max_concurrent:
            raise Exception(
                f"Maximum concurrent sessions ({self.max_concurrent}) reached"
            )
        session = AnthropicSession(session_id, self.client)
        self.active_sessions[session_id] = session
        logger.info(
            "Session created",
            session_id=session_id,
            active_sessions=len(self.active_sessions),
        )
        return session

    async def get_session(self, session_id: str) -> Optional[AnthropicSession]:
        """Get existing session."""
        return self.active_sessions.get(session_id)

    async def stop_session(self, session_id: str):
        """Remove a session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(
                "Session stopped",
                session_id=session_id,
                active_sessions=len(self.active_sessions),
            )

    async def cleanup_all(self):
        """Remove all sessions."""
        self.active_sessions.clear()
        logger.info("All sessions cleaned up")

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())


# Utility functions kept for backward compatibility
def create_project_directory(project_id: str) -> str:
    """Create project directory."""
    project_path = os.path.join(settings.project_root, project_id)
    os.makedirs(project_path, exist_ok=True)
    return project_path
