"""Server-Sent Events streaming utilities."""

import json
from typing import Dict, Any

import structlog

logger = structlog.get_logger()


class SSEFormatter:
    """Formats data for Server-Sent Events."""

    @staticmethod
    def format_event(data: Dict[str, Any]) -> str:
        """Format a JSON object as an SSE data line."""
        json_data = json.dumps(data, separators=(",", ":"))
        return f"data: {json_data}\n\n"

    @staticmethod
    def format_completion(data: str = "") -> str:
        """Format the [DONE] completion signal."""
        return "data: [DONE]\n\n"

    @staticmethod
    def format_error(error: str, error_type: str = "error") -> str:
        """Format an error as SSE."""
        error_data = {
            "error": {
                "message": error,
                "type": error_type,
                "code": "stream_error",
            }
        }
        return SSEFormatter.format_event(error_data)

    @staticmethod
    def format_heartbeat() -> str:
        """Format a heartbeat comment line."""
        return ": heartbeat\n\n"
