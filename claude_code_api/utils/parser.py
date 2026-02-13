"""Utility functions for content handling."""

from datetime import datetime
from typing import Optional


def sanitize_content(content: str) -> str:
    """Sanitize content for safe transmission."""
    if not content:
        return ""

    content = content.replace("\x00", "")
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    try:
        content.encode("utf-8")
    except UnicodeEncodeError:
        content = content.encode("utf-8", errors="replace").decode("utf-8")

    return content


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 chars per token)."""
    return max(1, len(text) // 4)


def format_timestamp(timestamp: Optional[str] = None) -> str:
    """Format timestamp for display."""
    if not timestamp:
        return datetime.utcnow().isoformat()
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.isoformat()
    except Exception:
        return timestamp
