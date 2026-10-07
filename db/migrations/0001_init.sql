-- =============================================================================
-- 0001_init.sql: Neurax initial schema
--
-- Source of truth for the data model. Never edit this file after it has been
-- applied anywhere; make schema changes in a new numbered migration instead.
--
-- Core idea: memories are never edited in place. A changed decision is a NEW
-- memory that supersedes the old one, so history is kept and agents only see
-- the current state by default.
-- =============================================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- Tracks which migrations have run. The migration runner inserts one row per file.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text        PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

-- Shared trigger: keeps updated_at current on every UPDATE.
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$;

-- -----------------------------------------------------------------------------
-- users
-- -----------------------------------------------------------------------------
CREATE TABLE users (
    id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    email        text        NOT NULL,
    display_name text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT users_email_format CHECK (position('@' IN email) > 1)
);

-- Case-insensitive uniqueness: "A@x.com" and "a@x.com" are the same user.
CREATE UNIQUE INDEX users_email_lower_uq ON users (lower(email));

CREATE TRIGGER users_set_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- -----------------------------------------------------------------------------
-- projects
-- -----------------------------------------------------------------------------
CREATE TABLE projects (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid        NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    slug        text        NOT NULL,   -- what agents pass as `project`, e.g. "neurax"
    name        text        NOT NULL,
    description text,
    archived_at timestamptz,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT projects_slug_format CHECK (slug ~ '^[a-z0-9][a-z0-9-]{0,62}$'),
    CONSTRAINT projects_user_slug_uq UNIQUE (user_id, slug),
    -- Lets memories reference (project, user) together, so a memory can never
    -- point at another user's project.
    CONSTRAINT projects_id_user_uq UNIQUE (id, user_id)
);

CREATE TRIGGER projects_set_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- -----------------------------------------------------------------------------
-- memories
-- -----------------------------------------------------------------------------
CREATE TABLE memories (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         uuid        NOT NULL,
    project_id      uuid        NOT NULL,

    type            text        NOT NULL,
    -- Normalized grouping key, e.g. "database", "auth/session-store".
    -- Supersession and contradiction checks work within a topic.
    topic           text        NOT NULL,
    content         text        NOT NULL,   -- the decision/fact itself, one statement
    rationale       text,                   -- why; optional but encouraged for decisions

    status          text        NOT NULL DEFAULT 'active',
    supersedes_id   uuid,                   -- the older memory this one replaces
    superseded_at   timestamptz,            -- set when THIS memory gets replaced

    pinned          boolean     NOT NULL DEFAULT false,
    source_tool     text        NOT NULL DEFAULT 'unknown',  -- "claude-code", "cursor", ...
    session_ref     text,                   -- optional pointer to the originating session

    -- Exact-duplicate detection (case and surrounding whitespace ignored).
    content_hash    text        GENERATED ALWAYS AS (md5(lower(btrim(content)))) STORED,

    -- NULL until the embedding worker fills it. bge-small-en-v1.5 = 384 dims.
    embedding       vector(384),
    embedding_model text,

    -- Keyword search document. Topic weighs most, then content, then rationale.
    search_tsv      tsvector    GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', translate(topic, '/-._', '    ')), 'A') ||
                        setweight(to_tsvector('english', content), 'B') ||
                        setweight(to_tsvector('english', coalesce(rationale, '')), 'C')
                    ) STORED,

    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT memories_type_valid      CHECK (type IN ('decision', 'constraint', 'fact', 'task')),
    CONSTRAINT memories_status_valid    CHECK (status IN ('active', 'superseded')),
    CONSTRAINT memories_topic_format    CHECK (topic ~ '^[a-z0-9][a-z0-9._/-]{0,79}$'),
    CONSTRAINT memories_content_len     CHECK (length(btrim(content)) BETWEEN 1 AND 4000),
    CONSTRAINT memories_rationale_len   CHECK (rationale IS NULL OR length(rationale) <= 4000),
    CONSTRAINT memories_source_len      CHECK (length(source_tool) BETWEEN 1 AND 64),
    CONSTRAINT memories_session_len     CHECK (session_ref IS NULL OR length(session_ref) <= 200),
    CONSTRAINT memories_no_self_supersede CHECK (supersedes_id IS NULL OR supersedes_id <> id),
    -- superseded <=> has a superseded_at timestamp
    CONSTRAINT memories_superseded_ts   CHECK ((status = 'superseded') = (superseded_at IS NOT NULL)),
    -- a replaced memory cannot stay pinned
    CONSTRAINT memories_no_pinned_superseded CHECK (NOT (pinned AND status = 'superseded')),
    -- embedding and the model that produced it are set together
    CONSTRAINT memories_embedding_model CHECK ((embedding IS NULL) = (embedding_model IS NULL)),

    -- Tenant safety: the project must belong to the same user.
    CONSTRAINT memories_project_fk FOREIGN KEY (project_id, user_id)
        REFERENCES projects (id, user_id) ON DELETE CASCADE,
    -- Target for the self-reference below.
    CONSTRAINT memories_tenant_uq UNIQUE (id, user_id, project_id),
    -- A memory can only supersede one in the same user AND project.
    -- If the old memory is hard-deleted, only supersedes_id is cleared.
    CONSTRAINT memories_supersedes_fk FOREIGN KEY (supersedes_id, user_id, project_id)
        REFERENCES memories (id, user_id, project_id) ON DELETE SET NULL (supersedes_id)
);

-- History stays a straight line: each memory can be superseded at most once.
CREATE UNIQUE INDEX memories_supersedes_once_uq
    ON memories (supersedes_id) WHERE supersedes_id IS NOT NULL;

-- No two active memories in a project with identical content.
CREATE UNIQUE INDEX memories_active_content_uq
    ON memories (project_id, content_hash) WHERE status = 'active';

-- Main listing path: a user's project, current items, newest first.
CREATE INDEX memories_user_project_status_idx
    ON memories (user_id, project_id, status, created_at DESC);

-- Topic lookups for supersession, contradiction checks and decision_history.
CREATE INDEX memories_project_topic_idx
    ON memories (project_id, topic, created_at DESC);

-- Pinned items are always included in context, so fetch them cheaply.
CREATE INDEX memories_pinned_idx
    ON memories (project_id) WHERE pinned;

-- Full-text search.
CREATE INDEX memories_search_tsv_idx
    ON memories USING gin (search_tsv);

-- Approximate nearest-neighbour search on cosine distance.
CREATE INDEX memories_embedding_hnsw_idx
    ON memories USING hnsw (embedding vector_cosine_ops);

CREATE TRIGGER memories_set_updated_at
    BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- -----------------------------------------------------------------------------
-- Guard: enforce "never edit, supersede instead" at the database level, so no
-- application bug can rewrite history.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION memories_guard_update() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    -- Ownership never moves.
    IF NEW.user_id <> OLD.user_id OR NEW.project_id <> OLD.project_id THEN
        RAISE EXCEPTION 'memory % cannot change user or project', OLD.id
            USING ERRCODE = 'check_violation';
    END IF;

    -- The statement itself is immutable. To change it, save a new memory and supersede.
    IF NEW.content   IS DISTINCT FROM OLD.content
       OR NEW.rationale IS DISTINCT FROM OLD.rationale
       OR NEW.topic     IS DISTINCT FROM OLD.topic
       OR NEW.type      IS DISTINCT FROM OLD.type THEN
        RAISE EXCEPTION 'memory % is immutable: save a new memory and supersede it instead', OLD.id
            USING ERRCODE = 'check_violation';
    END IF;

    -- No un-superseding: history only moves forward.
    IF OLD.status = 'superseded' AND NEW.status = 'active' THEN
        RAISE EXCEPTION 'memory % is superseded and cannot be reactivated', OLD.id
            USING ERRCODE = 'check_violation';
    END IF;

    -- supersedes_id may be set once (NULL -> value) or cleared by FK cleanup (-> NULL).
    IF OLD.supersedes_id IS NOT NULL
       AND NEW.supersedes_id IS NOT NULL
       AND NEW.supersedes_id <> OLD.supersedes_id THEN
        RAISE EXCEPTION 'memory % already supersedes %', OLD.id, OLD.supersedes_id
            USING ERRCODE = 'check_violation';
    END IF;

    -- Superseded rows are frozen except for re-embedding and FK cleanup.
    IF OLD.status = 'superseded' AND (
           NEW.pinned        IS DISTINCT FROM OLD.pinned
        OR NEW.superseded_at IS DISTINCT FROM OLD.superseded_at
        OR NEW.source_tool   IS DISTINCT FROM OLD.source_tool
        OR NEW.session_ref   IS DISTINCT FROM OLD.session_ref
        OR (NEW.supersedes_id IS NOT NULL AND OLD.supersedes_id IS NULL)
    ) THEN
        RAISE EXCEPTION 'memory % is superseded and read-only', OLD.id
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER memories_guard
    BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION memories_guard_update();

-- -----------------------------------------------------------------------------
-- neurax_supersede: the ONLY way application code should supersede a memory.
-- Atomic: marks the old memory superseded, links the new one to it, and moves
-- the pin across if the old one was pinned. Locks both rows to avoid races
-- between two agents saving at the same moment.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION neurax_supersede(p_user_id uuid, p_old_id uuid, p_new_id uuid)
RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    v_old memories%ROWTYPE;
    v_new memories%ROWTYPE;
BEGIN
    IF p_old_id = p_new_id THEN
        RAISE EXCEPTION 'a memory cannot supersede itself' USING ERRCODE = 'check_violation';
    END IF;

    -- Lock in id order so two concurrent calls cannot deadlock.
    PERFORM 1 FROM memories
     WHERE id IN (p_old_id, p_new_id) AND user_id = p_user_id
     ORDER BY id
       FOR UPDATE;

    SELECT * INTO v_old FROM memories WHERE id = p_old_id AND user_id = p_user_id;
    SELECT * INTO v_new FROM memories WHERE id = p_new_id AND user_id = p_user_id;

    IF v_old.id IS NULL OR v_new.id IS NULL THEN
        RAISE EXCEPTION 'memory not found' USING ERRCODE = 'no_data_found';
    END IF;
    IF v_old.project_id <> v_new.project_id THEN
        RAISE EXCEPTION 'memories belong to different projects' USING ERRCODE = 'check_violation';
    END IF;
    IF v_old.status <> 'active' THEN
        RAISE EXCEPTION 'memory % is already superseded', p_old_id USING ERRCODE = 'check_violation';
    END IF;
    IF v_new.status <> 'active' THEN
        RAISE EXCEPTION 'memory % is superseded and cannot replace another', p_new_id
            USING ERRCODE = 'check_violation';
    END IF;
    IF v_new.supersedes_id IS NOT NULL THEN
        RAISE EXCEPTION 'memory % already supersedes %', p_new_id, v_new.supersedes_id
            USING ERRCODE = 'check_violation';
    END IF;

    UPDATE memories
       SET status = 'superseded', superseded_at = now(), pinned = false
     WHERE id = p_old_id;

    UPDATE memories
       SET supersedes_id = p_old_id, pinned = (v_new.pinned OR v_old.pinned)
     WHERE id = p_new_id;
END;
$$;

INSERT INTO schema_migrations (version) VALUES ('0001_init');

COMMIT;
