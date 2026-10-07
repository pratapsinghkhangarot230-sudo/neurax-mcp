"""Settings from environment variables only. Defaults match docs/ARCHITECTURE.md section 6."""

import os

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", "postgresql://neurax:neurax@localhost:5432/neurax"
)
LOG_LEVEL: str = os.environ.get("NEURAX_LOG_LEVEL", "INFO")
