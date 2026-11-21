-- =============================================================================
-- Migração v002 — Chunks Estruturais, Índice Lexical SQLite FTS5 e Registro Vetorial
-- =============================================================================

CREATE TABLE IF NOT EXISTS normative_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    unit_id TEXT NOT NULL,
    authority TEXT NOT NULL,
    document_type TEXT NOT NULL,
    regulation_number TEXT NOT NULL,
    amendment_number TEXT NOT NULL,
    version_status TEXT NOT NULL,
    section_number TEXT NOT NULL,
    subsection_number TEXT NOT NULL DEFAULT '',
    paragraph_number TEXT NOT NULL DEFAULT '',
    item_number TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    subpart TEXT NOT NULL DEFAULT '',
    start_page INTEGER NOT NULL CHECK (start_page >= 1),
    end_page INTEGER NOT NULL CHECK (end_page >= start_page),
    pdf_pages_json TEXT NOT NULL DEFAULT '[]',
    printed_pages_json TEXT NOT NULL DEFAULT '[]',
    chunk_index INTEGER NOT NULL DEFAULT 0 CHECK (chunk_index >= 0),
    original_text TEXT NOT NULL,
    search_text TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    validation_status TEXT NOT NULL DEFAULT 'pending_review',
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dimension INTEGER NOT NULL DEFAULT 0,
    indexed_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES regulatory_documents (document_id) ON DELETE CASCADE,
    FOREIGN KEY (version_id) REFERENCES document_versions (version_id) ON DELETE CASCADE,
    FOREIGN KEY (unit_id) REFERENCES normative_units (unit_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_version_section
    ON normative_chunks (version_id, section_number, chunk_index);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_reg_amend
    ON normative_chunks (document_id, regulation_number, amendment_number, version_status);

CREATE INDEX IF NOT EXISTS idx_chunks_unit
    ON normative_chunks (unit_id);

CREATE VIRTUAL TABLE IF NOT EXISTS normative_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    version_id UNINDEXED,
    regulation_number,
    amendment_number,
    section_number,
    subsection_number,
    paragraph_number,
    item_number,
    title,
    search_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS vector_index_registry (
    collection_name TEXT NOT NULL,
    version_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension >= 0),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    indexed_at TEXT NOT NULL,
    PRIMARY KEY (collection_name, version_id),
    FOREIGN KEY (document_id) REFERENCES regulatory_documents (document_id) ON DELETE CASCADE,
    FOREIGN KEY (version_id) REFERENCES document_versions (version_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vector_registry_collection
    ON vector_index_registry (collection_name, document_id);
