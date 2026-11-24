-- =============================================================================
-- Migração v004 — Metadados Estruturais e Geométricos de Tabelas/Formulários
-- =============================================================================

ALTER TABLE normative_tables
    ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';
