-- =============================================================================
-- Migração v001 — Catálogo Multidocumento Regulatório, Páginas, Unidades e Revisões
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regulatory_documents (
    document_id TEXT PRIMARY KEY,
    authority TEXT NOT NULL,
    document_type TEXT NOT NULL,
    regulation_number TEXT NOT NULL,
    document_title TEXT NOT NULL,
    document_status TEXT NOT NULL DEFAULT 'active',
    version_status TEXT NOT NULL DEFAULT 'unknown',
    amendment_number TEXT NOT NULL DEFAULT '',
    publication_date TEXT,
    effective_date TEXT,
    expiration_date TEXT,
    official_url TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    pdf_filename TEXT NOT NULL DEFAULT '',
    file_hash TEXT NOT NULL DEFAULT '',
    imported_at TEXT,
    last_verified_at TEXT,
    subject_categories_json TEXT NOT NULL DEFAULT '[]',
    document_description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (authority, document_type, regulation_number)
);

CREATE TABLE IF NOT EXISTS document_versions (
    version_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    amendment_number TEXT NOT NULL,
    resolution_number TEXT NOT NULL DEFAULT '',
    publication_date TEXT,
    effective_date TEXT,
    expiration_date TEXT,
    version_status TEXT NOT NULL DEFAULT 'unknown',
    official_url TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    pdf_filename TEXT NOT NULL DEFAULT '',
    pdf_storage_path TEXT NOT NULL DEFAULT '',
    source_hash TEXT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0 CHECK (page_count >= 0),
    imported_at TEXT,
    last_verified_at TEXT,
    FOREIGN KEY (document_id) REFERENCES regulatory_documents (document_id) ON DELETE CASCADE,
    UNIQUE (document_id, amendment_number, source_hash)
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    run_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version_id TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    source_hash TEXT NOT NULL DEFAULT '',
    pages_extracted INTEGER NOT NULL DEFAULT 0,
    units_created INTEGER NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (document_id) REFERENCES regulatory_documents (document_id) ON DELETE CASCADE,
    FOREIGN KEY (version_id) REFERENCES document_versions (version_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS regulatory_pages (
    page_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    pdf_page_number INTEGER NOT NULL CHECK (pdf_page_number >= 1),
    printed_page_number TEXT NOT NULL DEFAULT '',
    original_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    extraction_quality_json TEXT NOT NULL DEFAULT '{}',
    coordinates_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (version_id) REFERENCES document_versions (version_id) ON DELETE CASCADE,
    UNIQUE (version_id, pdf_page_number)
);

CREATE TABLE IF NOT EXISTS normative_units (
    unit_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    parent_unit_id TEXT,
    unit_type TEXT NOT NULL,
    section_number TEXT NOT NULL,
    subsection_number TEXT NOT NULL DEFAULT '',
    paragraph_number TEXT NOT NULL DEFAULT '',
    item_number TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    subpart TEXT NOT NULL DEFAULT '',
    original_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    start_page INTEGER NOT NULL CHECK (start_page >= 1),
    end_page INTEGER NOT NULL CHECK (end_page >= start_page),
    order_index INTEGER NOT NULL DEFAULT 0,
    structure_status TEXT NOT NULL DEFAULT 'parsed',
    validation_status TEXT NOT NULL DEFAULT 'pending_review',
    content_hash TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (version_id) REFERENCES document_versions (version_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS normative_unit_pages (
    unit_id TEXT NOT NULL,
    page_id TEXT NOT NULL,
    sequence_order INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (unit_id, page_id),
    FOREIGN KEY (unit_id) REFERENCES normative_units (unit_id) ON DELETE CASCADE,
    FOREIGN KEY (page_id) REFERENCES regulatory_pages (page_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS normative_tables (
    table_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    unit_id TEXT,
    pdf_page_number INTEGER NOT NULL CHECK (pdf_page_number >= 1),
    caption TEXT NOT NULL DEFAULT '',
    rows_json TEXT NOT NULL,
    FOREIGN KEY (version_id) REFERENCES document_versions (version_id) ON DELETE CASCADE,
    FOREIGN KEY (unit_id) REFERENCES normative_units (unit_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS normative_references (
    reference_id TEXT PRIMARY KEY,
    unit_id TEXT NOT NULL,
    target_regulation TEXT NOT NULL,
    target_section TEXT NOT NULL DEFAULT '',
    raw_citation TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (unit_id) REFERENCES normative_units (unit_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS validation_records (
    review_id TEXT PRIMARY KEY,
    page_id TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    decision TEXT NOT NULL,
    comments TEXT NOT NULL DEFAULT '',
    source_hash TEXT NOT NULL,
    FOREIGN KEY (page_id) REFERENCES regulatory_pages (page_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_type_number
    ON regulatory_documents (authority, document_type, regulation_number);

CREATE INDEX IF NOT EXISTS idx_versions_document_status
    ON document_versions (document_id, version_status);

CREATE INDEX IF NOT EXISTS idx_pages_version_status
    ON regulatory_pages (version_id, validation_status);

CREATE INDEX IF NOT EXISTS idx_units_version_section
    ON normative_units (version_id, section_number, order_index);

CREATE INDEX IF NOT EXISTS idx_reviews_page_date
    ON validation_records (page_id, reviewed_at);
