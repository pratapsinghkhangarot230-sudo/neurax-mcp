# One-command setup: Postgres in Docker, migrations, embedding model, then the MCP command.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Email you@example.com]
#
# The email is your Neurax identity; defaults to `git config user.email`.
param([string]$Email)
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

function Fail($msg) { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }

# Native commands don't throw on a non-zero exit code; stop on the first failure.
function Run([string]$label, [scriptblock]$cmd) {
    Write-Host "==> $label"
    & $cmd
    if ($LASTEXITCODE -ne 0) { Fail "$label failed (exit code $LASTEXITCODE)" }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail "Docker not found. Install Docker Desktop: https://docs.docker.com/desktop/"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "uv not found. Install uv: https://docs.astral.sh/uv/getting-started/installation/"
}
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail "Docker is installed but not running. Start Docker Desktop and run this again." }

Run "Installing Python dependencies" { uv sync }
Run "Starting Postgres (docker compose)" { docker compose up -d --wait }
Run "Applying migrations" { uv run neurax-migrate }
Run "Checking the database" { uv run python -m neurax.health }
Run "Downloading the embedding model (about 70 MB, once)" {
    uv run python -c "from neurax import embeddings; embeddings.embed(['warmup'])"
}

if (-not $Email) { $Email = (git config user.email 2>$null) }
if (-not $Email) { $Email = "you@example.com" }

Write-Host ""
Write-Host "Neurax is ready. Register it with Claude Code (once, for every project):"
Write-Host ""
Write-Host "  claude mcp add --scope user --env NEURAX_USER_EMAIL=$Email --transport stdio neurax -- uv --directory `"$Repo`" run neurax-server"
Write-Host ""
Write-Host "Cursor and Claude Desktop: see `"Connect your agent`" in README.md."
