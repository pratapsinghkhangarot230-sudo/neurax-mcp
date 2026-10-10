#!/usr/bin/env bash
# One-command setup: Postgres in Docker, migrations, embedding model, then the MCP command.
#
#   scripts/install.sh [you@example.com]
#
# The email is your Neurax identity; defaults to `git config user.email`.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd -W 2>/dev/null || pwd)"  # pwd -W: Windows path under Git Bash

fail() { echo "error: $*" >&2; exit 1; }

command -v docker >/dev/null || fail "Docker not found. Install Docker: https://docs.docker.com/get-docker/"
command -v uv >/dev/null || fail "uv not found. Install uv: https://docs.astral.sh/uv/getting-started/installation/"
docker info >/dev/null 2>&1 || fail "Docker is installed but not running. Start Docker and run this again."

echo "==> Installing Python dependencies"
uv sync
echo "==> Starting Postgres (docker compose)"
docker compose up -d --wait
echo "==> Applying migrations"
uv run neurax-migrate
uv run python -m neurax.health
echo "==> Downloading the embedding model (about 70 MB, once)"
uv run python -c "from neurax import embeddings; embeddings.embed(['warmup'])"

EMAIL="${1:-$(git config user.email 2>/dev/null || true)}"
EMAIL="${EMAIL:-you@example.com}"

cat <<EOF

Neurax is ready. Register it with Claude Code (once, for every project):

  claude mcp add --scope user --env NEURAX_USER_EMAIL=$EMAIL --transport stdio neurax -- uv --directory "$REPO" run neurax-server

Cursor and Claude Desktop: see "Connect your agent" in README.md.
EOF
