"""Settings from environment variables only. Defaults match docs/ARCHITECTURE.md section 6."""

import os
from pathlib import Path


def _default_model_cache() -> str:
    # A fixed location: fastembed's default lives under the temp dir, which Windows may
    # clean, forcing a re-download at server start (slow enough to time out MCP startup).
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return str(Path(os.environ["LOCALAPPDATA"]) / "neurax" / "models")
    return str(Path.home() / ".cache" / "neurax" / "models")


DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://neurax:neurax@127.0.0.1:5432/neurax"
)
LOG_LEVEL: str = os.environ.get("NEURAX_LOG_LEVEL", "INFO")
# Required by the MCP server (Phase 2); memory.py takes user_id explicitly.
USER_EMAIL: str | None = os.environ.get("NEURAX_USER_EMAIL")
EMBED_MODEL: str = os.environ.get("NEURAX_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_CACHE: str = os.environ.get("NEURAX_MODEL_CACHE") or _default_model_cache()
CONFLICT_THRESHOLD: float = float(os.environ.get("NEURAX_CONFLICT_THRESHOLD", "0.68"))

# Ranking weights and recency half-life (ARCHITECTURE section 4).
W_VEC: float = float(os.environ.get("NEURAX_W_VEC", "0.55"))
W_TEXT: float = float(os.environ.get("NEURAX_W_TEXT", "0.25"))
W_RECENT: float = float(os.environ.get("NEURAX_W_RECENT", "0.10"))
W_PIN: float = float(os.environ.get("NEURAX_W_PIN", "0.10"))
HALF_LIFE_DAYS: float = float(os.environ.get("NEURAX_HALF_LIFE_DAYS", "30"))
if HALF_LIFE_DAYS <= 0:
    # Fail at startup, not as a divide-by-zero inside the search SQL.
    raise ValueError("NEURAX_HALF_LIFE_DAYS must be > 0")
