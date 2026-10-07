"""Settings from environment variables only. Defaults match docs/ARCHITECTURE.md section 6."""

import os

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://neurax:neurax@127.0.0.1:5432/neurax"
)
LOG_LEVEL: str = os.environ.get("NEURAX_LOG_LEVEL", "INFO")
# Required by the MCP server (Phase 2); memory.py takes user_id explicitly.
USER_EMAIL: str | None = os.environ.get("NEURAX_USER_EMAIL")
EMBED_MODEL: str = os.environ.get("NEURAX_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CONFLICT_THRESHOLD: float = float(os.environ.get("NEURAX_CONFLICT_THRESHOLD", "0.80"))
