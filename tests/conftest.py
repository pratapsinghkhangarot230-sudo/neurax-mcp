"""Integration test setup: a real Postgres database `neurax_test`, a fresh user per test."""

import os
import uuid
from uuid import UUID

import psycopg
import pytest

TEST_DATABASE_URL = os.environ.get(
    "NEURAX_TEST_DATABASE_URL", "postgresql://neurax:neurax@127.0.0.1:5432/neurax_test"
)
# Point neurax at the test database before any neurax module reads config.
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

from neurax import memory, migrate  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def test_database() -> None:
    dbname = psycopg.conninfo.conninfo_to_dict(TEST_DATABASE_URL)["dbname"]
    assert str(dbname).endswith("_test"), "tests must run against a *_test database"
    migrate.create_database()  # created once, never dropped by the tests
    migrate.migrate()


@pytest.fixture
def user_id() -> UUID:
    # A fresh user per test: tenancy isolates tests, so nothing needs deleting.
    return memory.ensure_user(f"test-{uuid.uuid4().hex}@example.com")


@pytest.fixture
def project(user_id: UUID) -> str:
    memory.create_project(user_id, "demo", "Demo project")
    return "demo"
