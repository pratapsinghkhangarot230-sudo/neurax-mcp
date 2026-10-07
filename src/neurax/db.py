"""Connection pool and transaction helper."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

import psycopg
from psycopg_pool import ConnectionPool

from neurax import config


@cache
def get_pool() -> ConnectionPool:
    """One pool per process, opened on first use."""
    return ConnectionPool(config.DATABASE_URL, min_size=1, max_size=5, open=True)


@contextmanager
def transaction() -> Iterator[psycopg.Connection]:
    """Yield a pooled connection inside one transaction: commit on success, roll back on error."""
    with get_pool().connection() as conn, conn.transaction():
        yield conn
