-- =============================================================================
-- Schema invariant tests. Run against a database with migrations applied:
--   psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f db/tests/test_invariants.sql
-- Everything runs inside a transaction that is rolled back, so it leaves no data.
-- Prints "ALL INVARIANT TESTS PASSED" on success; any failure aborts with an error.
-- =============================================================================

BEGIN;

-- Helper: asserts that a statement fails with an error message containing `expected`.
CREATE FUNCTION pg_temp.expect_error(stmt text, expected text) RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    BEGIN
        EXECUTE stmt;
    EXCEPTION WHEN OTHERS THEN
        IF position(expected IN SQLERRM) = 0 THEN
            RAISE EXCEPTION 'FAIL: expected error containing "%", got "%" for: %', expected, SQLERRM, stmt;
        END IF;
        RETURN;
    END;
    RAISE EXCEPTION 'FAIL: expected an error containing "%" but statement succeeded: %', expected, stmt;
END;
$$;

DO $$
DECLARE
    u1 uuid; u2 uuid;
    p1 uuid; p2 uuid; p_other uuid;
    m_redis uuid; m_pg uuid; m_mysql uuid; m_other uuid; m_cross uuid;
    v_status text; v_pinned boolean; v_chain int;
BEGIN
    -- Setup: two users, each with a project.
    INSERT INTO users (email) VALUES ('pratap@example.com') RETURNING id INTO u1;
    INSERT INTO users (email) VALUES ('someone@example.com') RETURNING id INTO u2;
    INSERT INTO projects (user_id, slug, name) VALUES (u1, 'neurax', 'Neurax') RETURNING id INTO p1;
    INSERT INTO projects (user_id, slug, name) VALUES (u1, 'datum', 'Datum') RETURNING id INTO p2;
    INSERT INTO projects (user_id, slug, name) VALUES (u2, 'neurax', 'Their Neurax') RETURNING id INTO p_other;

    -- 1. Emails are unique case-insensitively.
    PERFORM pg_temp.expect_error(
        $q$INSERT INTO users (email) VALUES ('PRATAP@example.com')$q$, 'users_email_lower_uq');

    -- 2. A user cannot attach a memory to another user's project.
    PERFORM pg_temp.expect_error(format(
        $q$INSERT INTO memories (user_id, project_id, type, topic, content)
           VALUES (%L, %L, 'decision', 'database', 'x')$q$, u1, p_other), 'memories_project_fk');

    -- 3. Topic must be a normalized slug.
    PERFORM pg_temp.expect_error(format(
        $q$INSERT INTO memories (user_id, project_id, type, topic, content)
           VALUES (%L, %L, 'decision', 'Database Choice', 'x')$q$, u1, p1), 'memories_topic_format');

    -- 4. Type must be one of the allowed values.
    PERFORM pg_temp.expect_error(format(
        $q$INSERT INTO memories (user_id, project_id, type, topic, content)
           VALUES (%L, %L, 'opinion', 'database', 'x')$q$, u1, p1), 'memories_type_valid');

    -- Real data: a decision, later replaced, later replaced again.
    INSERT INTO memories (user_id, project_id, type, topic, content, rationale, pinned, source_tool)
    VALUES (u1, p1, 'decision', 'database', 'Use Redis for memory storage', 'fast', true, 'claude-code')
    RETURNING id INTO m_redis;

    -- 5. Exact duplicate active content in the same project is rejected...
    PERFORM pg_temp.expect_error(format(
        $q$INSERT INTO memories (user_id, project_id, type, topic, content)
           VALUES (%L, %L, 'decision', 'database', '  use redis for memory storage ')$q$, u1, p1),
        'memories_active_content_uq');
    -- ...but the same text in another project is fine.
    INSERT INTO memories (user_id, project_id, type, topic, content)
    VALUES (u1, p2, 'decision', 'database', 'Use Redis for memory storage') RETURNING id INTO m_other;

    -- 6. Content cannot be edited in place.
    PERFORM pg_temp.expect_error(format(
        $q$UPDATE memories SET content = 'Use Postgres' WHERE id = %L$q$, m_redis), 'immutable');

    -- 7. Supersede: Redis -> Postgres. Pin should move to the new memory.
    INSERT INTO memories (user_id, project_id, type, topic, content, rationale, source_tool)
    VALUES (u1, p1, 'decision', 'database', 'Use Postgres with pgvector', 'one database for everything', 'cursor')
    RETURNING id INTO m_pg;
    PERFORM neurax_supersede(u1, m_redis, m_pg);

    SELECT status, pinned INTO v_status, v_pinned FROM memories WHERE id = m_redis;
    IF v_status <> 'superseded' OR v_pinned THEN
        RAISE EXCEPTION 'FAIL: old memory should be superseded and unpinned (got %, %)', v_status, v_pinned;
    END IF;
    SELECT pinned INTO v_pinned FROM memories WHERE id = m_pg;
    IF NOT v_pinned THEN
        RAISE EXCEPTION 'FAIL: pin should move to the new memory';
    END IF;

    -- 8. A superseded memory cannot be superseded again (history stays linear).
    INSERT INTO memories (user_id, project_id, type, topic, content)
    VALUES (u1, p1, 'decision', 'database', 'Use MySQL') RETURNING id INTO m_mysql;
    PERFORM pg_temp.expect_error(format(
        $q$SELECT neurax_supersede(%L, %L, %L)$q$, u1, m_redis, m_mysql), 'already superseded');

    -- 9. A superseded memory cannot be reactivated or re-pinned.
    PERFORM pg_temp.expect_error(format(
        $q$UPDATE memories SET status = 'active', superseded_at = NULL WHERE id = %L$q$, m_redis),
        'cannot be reactivated');
    PERFORM pg_temp.expect_error(format(
        $q$UPDATE memories SET pinned = true WHERE id = %L$q$, m_redis), 'read-only');

    -- 10. Cannot supersede across projects.
    PERFORM pg_temp.expect_error(format(
        $q$SELECT neurax_supersede(%L, %L, %L)$q$, u1, m_other, m_mysql), 'different projects');

    -- 11. Another user cannot supersede your memories (they simply are not found).
    PERFORM pg_temp.expect_error(format(
        $q$SELECT neurax_supersede(%L, %L, %L)$q$, u2, m_pg, m_mysql), 'memory not found');

    -- 12. Self-supersede is rejected.
    PERFORM pg_temp.expect_error(format(
        $q$SELECT neurax_supersede(%L, %L, %L)$q$, u1, m_pg, m_pg), 'cannot supersede itself');

    -- 13. Second hop works: Postgres -> MySQL. Chain should be 3 long.
    PERFORM neurax_supersede(u1, m_pg, m_mysql);
    WITH RECURSIVE chain AS (
        SELECT id, supersedes_id FROM memories WHERE id = m_mysql
        UNION ALL
        SELECT m.id, m.supersedes_id FROM memories m JOIN chain c ON m.id = c.supersedes_id
    )
    SELECT count(*) INTO v_chain FROM chain;
    IF v_chain <> 3 THEN
        RAISE EXCEPTION 'FAIL: expected decision chain of 3, got %', v_chain;
    END IF;

    -- 14. Only one active memory in the topic now.
    IF (SELECT count(*) FROM memories WHERE project_id = p1 AND topic = 'database' AND status = 'active') <> 1 THEN
        RAISE EXCEPTION 'FAIL: expected exactly one active memory for topic database';
    END IF;

    -- 15. Full-text search matches on topic and content.
    IF NOT EXISTS (SELECT 1 FROM memories
                    WHERE id = m_pg AND search_tsv @@ plainto_tsquery('english', 'pgvector database')) THEN
        RAISE EXCEPTION 'FAIL: full-text search did not match';
    END IF;

    -- 16. Embedding and embedding_model must be set together.
    PERFORM pg_temp.expect_error(format(
        $q$UPDATE memories SET embedding_model = 'bge-small-en-v1.5' WHERE id = %L$q$, m_mysql),
        'memories_embedding_model');

    -- 17. Deleting a project removes its memories.
    DELETE FROM projects WHERE id = p2;
    IF EXISTS (SELECT 1 FROM memories WHERE id = m_other) THEN
        RAISE EXCEPTION 'FAIL: project delete should cascade to memories';
    END IF;

    RAISE NOTICE 'ALL INVARIANT TESTS PASSED';
END;
$$;

ROLLBACK;
