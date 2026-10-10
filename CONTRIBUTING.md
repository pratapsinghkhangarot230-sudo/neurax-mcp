# Contributing to Neurax

Thanks for helping. Neurax is small on purpose; please read this before opening a PR.

## Setup

Needs Docker and [uv](https://docs.astral.sh/uv/).

```bash
scripts/install.sh            # or scripts\install.ps1 on Windows
```

## Before you open a PR

```bash
uv run ruff check
uv run ruff format --check
uv run mypy src tests examples
uv run pytest                 # integration tests against a real Postgres (neurax_test)
```

CI runs the same checks plus `db/tests/test_invariants.sql`.

## Rules

- **Scope.** v1 is locked (see the table in `CLAUDE.md`). New ideas go in `PARKING_LOT.md` first,
  with one line on why they matter. They get built after launch if users ask for them.
- **Plain SQL, no ORM.** Parameterized queries only (`%s`), and every query on `projects` or
  `memories` filters by `user_id`.
- **Migrations are never edited after they are applied.** Schema changes go in a new numbered
  file in `db/migrations/`. Open an issue first.
- **Memories are immutable.** A changed decision is a new memory plus `neurax_supersede()`.
- **Bug fixes come with a test** that fails without the fix.
- **No new dependencies** without discussing them in an issue first.
- Small commits with conventional messages: `feat:`, `fix:`, `docs:`, `test:`, `chore:`.

Architecture, tool contracts and invariants: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

By contributing you agree that your work is licensed under the Apache License 2.0.
