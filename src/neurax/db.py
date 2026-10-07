"""Connection pool and transaction helper."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import cast

import psycopg
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from neurax import config

# Pooled connections return rows as dicts: maps straight to JSON tool output.
Conn = psycopg.Connection[DictRow]


@cache
def get_pool() -> ConnectionPool:
    """One pool per process, opened on first use."""
    return ConnectionPool(
        config.DATABASE_URL,
        min_size=1,
        max_size=5,
        open=True,
        kwargs={"row_factory": dict_row},
    )


@contextmanager
def transaction() -> Iterator[Conn]:
    """Yield a pooled connection inside one transaction: commit on success, roll back on error."""
    with get_pool().connection() as conn, conn.transaction():
        yield cast(Conn, conn)
