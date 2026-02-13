"""Configuration management for Claude Code API Gateway."""

import os
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    api_title: str = "Claude Code API Gateway"
    api_version: str = "1.0.0"
    api_description: str = "OpenAI-compatible API for Claude Code"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Authentication
    api_keys: List[str] = Field(default_factory=list)
    require_auth: bool = False

    @field_validator('api_keys', mode='before')
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(',') if x.strip()]
        return v or []

    # Anthropic Configuration
    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-4-5-20250929"
    max_concurrent_sessions: int = 10
    session_timeout_minutes: int = 30

    # Project Configuration
    project_root: str = "/tmp/claude_projects"
    max_project_size_mb: int = 1000
    cleanup_interval_minutes: int = 60

    # Database Configuration
    database_url: str = "sqlite:///./claude_api.db"

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"

    # CORS Configuration
    allowed_origins: List[str] = Field(default=["*"])
    allowed_methods: List[str] = Field(default=["*"])
    allowed_headers: List[str] = Field(default=["*"])

    @field_validator('allowed_origins', 'allowed_methods', 'allowed_headers', mode='before')
    def parse_cors_lists(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(',') if x.strip()]
        return v or ["*"]

    # Rate Limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 10

    # Streaming Configuration
    streaming_chunk_size: int = 1024
    streaming_timeout_seconds: int = 300

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create global settings instance
settings = Settings()

# Ensure project root exists
os.makedirs(settings.project_root, exist_ok=True)
