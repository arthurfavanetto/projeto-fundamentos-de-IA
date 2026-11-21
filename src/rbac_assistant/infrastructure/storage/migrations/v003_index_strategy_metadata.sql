-- =============================================================================
-- Migração v003 — Metadado de Estratégia de Instrução no Registro Vetorial
-- =============================================================================

ALTER TABLE vector_index_registry
    ADD COLUMN instruction_strategy_version TEXT NOT NULL DEFAULT '';
