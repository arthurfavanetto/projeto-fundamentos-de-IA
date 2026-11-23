# Estado Atual do Projeto — RBAC AI Assistant

> **Referência de código:** `main` = `d1d6af150b0a20292ae7f91e33da0b91146573ca` (Etapas 0, 1, 2 e 3 integradas; PRs #4 a #9 mergeados).
>
> **Domínio:** Regulamentos Brasileiros da Aviação Civil (**RBAC — ANAC**; nunca *Role-Based Access Control*).

---

## 1. Arquitetura Consolidada (Hexagonal / Clean Architecture)

```text
src/rbac_assistant/
├── config/           # AppSettings (Pydantic Settings + YAML + RBAC_DISABLE_DOTENV), logging_config (stdout/stderr)
├── domain/           # Entidades normativas, exceções de domínio e ports (Protocol) — zero dependências externas
├── application/      # CatalogService, IngestionService, ValidationService, IndexingService, NormativeChatService e use cases
├── infrastructure/
│   ├── llm/          # GeminiClientAdapter + GeminiEmbeddingAdapter (google-genai==3.6.1, política única Free Tier)
│   ├── pdf/          # PdfPlumberExtractorAdapter, AnacRbacStructureStrategy, GenericStructureStrategy, TextQualityChecker, TesseractOcrAdapter
│   ├── sources/      # LocalFileSourceAdapter, AnacHttpSourceAdapter
│   ├── storage/      # SqliteDocumentRepository (SSOT ACID + WAL + FTS5 + suporte :memory:), LocalFileDocumentStore
│   └── vector_db/    # ChromaVectorStore (ChromaDB 1.5.9, guarda de compatibilidade modelo/dimensão/estratégia)
├── rag/
│   ├── ingestion/    # StructuralNormativeChunker
│   ├── retrieval/    # SectionResolver, HybridNormativeRetriever (Estrutural + FTS5 + ChromaDB → RRF + Reranker + Poda)
│   ├── generation/   # PromptTemplateStore, CitationValidator, NormativeResponseFormatter
│   └── evaluation/   # RagEvaluator (40 consultas sintéticas, 4 modos, 19 categorias, ablações, varredura de limiar)
├── security/         # PromptGuard (injeção direta/indireta), secret_redactor (mascaramento de tokens/chaves)
├── interfaces/
│   └── telegram/     # Placeholder (__init__.py) — Etapa 4 ainda NÃO implementada
├── healthcheck.py    # CLI de healthcheck Docker (ChromaDB heartbeat + escrita nos volumes)
└── main.py           # Bootstrap da Fundação Etapa 1 (aguarda SIGTERM/SIGINT; ainda NÃO inicia bot Telegram)
```

---

## 2. Matriz de Componentes (Implementado vs. Preparado vs. Pendente)

### Implementado e validado por testes automatizados (93 testes, 91% cobertura, CI `45068432151` verde)
- **Persistência SSOT e Lexical:** `SqliteDocumentRepository` (SQLite 3 WAL/ACID em `/app/data/documents/catalog.sqlite3`, suporte a `:memory:` em testes, migrações `v001`/`v002` com `normative_chunks_fts` FTS5).
- **Ingestão e Validação Regulatória:** extração nativa `pdfplumber`, dupla paginação (PDF + impressa), parser hierárquico ANAC/genérico, política híbrida `review_on_demand` com bloqueio estrito de OCR não revisado e páginas reprovadas.
- **Motor Vetorial e Gemini Free Tier:** `ChromaVectorStore` (`chromadb==1.5.9`, coleção `regulatory_chunks_ge2_768_v1`), `GeminiClientAdapter` com política única `gemini-4-hydrogen` + `gemini-embedding-2` (`768d`, `GEMINI_ALLOW_PAID_MODELS=false`, sem fallback silencioso).
- **RAG Híbrido (4 modos):** `structural_only`, `lexical_fts5_only`, `vector_only` e `hybrid_rrf`, com dedup N+1 por requisição na hidratação SSOT, `DeterministicNormativeReranker`, poda adaptativa e degradação controlada sem Gemini/ChromaDB.
- **Segurança e Fidelidade:** `PromptGuard`, `CitationValidator` (5 dimensões), `NormativeResponseFormatter` (4 blocos separados: `TEXTO NORMATIVO`, `EXPLICAÇÃO`, `REFERÊNCIA`, `OBSERVAÇÃO`), `secret_redactor`, isolamento `.env` (`RBAC_DISABLE_DOTENV=1`) e bloqueio de chamadas Gemini reais em testes/subprocessos (`RBAC_BLOCK_REAL_GEMINI_API=1`).
- **Avaliação e Cache de Benchmark:** `scripts/evaluate_rag.py` com `InMemoryQueryEmbeddingCache` (`1.286` → `46` chamadas HTTP simuladas no modo `compare`, zero persistência em disco).
- **CLIs operacionais:** `scripts/ingest_regulation.py`, `scripts/inspect_document.py`, `scripts/validate_extraction.py`, `scripts/index_document.py`, `scripts/query_assistant.py`, `scripts/evaluate_rag.py`, `scripts/verify_chroma_persistence.py`, `scripts/verify_sqlite_persistence.py`.
- **Qualidade e Paralelismo:** Ruff, MyPy strict (`82` arquivos) e `pytest-xdist` (`3.8.0`) com cobertura combinada idêntica.

### Implementado no código, pendente de revalidação local no Docker Windows pós-incidente
- Stack Docker Compose (`rbac-chromadb` + `rbac-bot`, UID/GID `10001`, rede interna `rbac_internal_net`, volumes novos `chroma_data`, `regulatory_documents`, `audit_logs`) — validada no CI Linux (`45068432151`), aguardando liberação do disco `/dev/sdd` no WSL e reinicialização do Docker Desktop no Windows.

### Preparado (configuração/arquitetura), mas NÃO implementado
- **Interface Telegram (Etapa 4):** existem apenas `TelegramConfig`, `AppSettings.telegram_bot_token`, `require_telegram_bot_token()`, mascaramento de tokens em `secret_redactor.py`, template `config/access_control.example.yaml`, ADR-03 e placeholder `src/rbac_assistant/interfaces/telegram/__init__.py`.
- **Ausentes (Telegram):** biblioteca de bot Telegram em `pyproject.toml`, Long Polling, handlers/comandos (`/start`, `/help`, `/status`, `/rbac`, `/buscar`, `/explicar`, `/documentos`, `/versao`, texto livre), `AccessControl` (`deny_by_default`), `MessageSplitter` (>4.096 chars), montagem de `config/access_control.yaml` em `docker-compose.yml` e integração no `src/rbac_assistant/main.py`.
- **Auditoria persistente em `/app/logs` (`AuditPort`):** [`src/rbac_assistant/domain/ports/audit_port.py`](../src/rbac_assistant/domain/ports/audit_port.py) é atualmente apenas um contrato `Protocol` sem adaptador concreto gravando JSONL em `/app/logs` (o volume `audit_logs` existe e tem escrita validada no startup/healthcheck; trilhas de revisão humana de páginas já são persistidas em `/app/data/documents/reviews` e SQLite `page_reviews`).
- **Rotação explícita de logs no Docker Compose:** `docker-compose.yml` ainda não declara `logging: {driver: json-file, options: {max-size: "10m", max-file: "5"}}` (usa padrão do Docker Desktop até PR autorizado).

### Pendente (operação / próximas etapas)
1. Liberação específica de `/dev/sdd` no WSL (recuperação forense VHDX encerrada definitivamente) e reconstrução limpa dos 3 volumes Docker novos/vazios no host Windows 11.
2. Validação local no Windows (Python 3.12 + testes + Docker healthchecks + persistência + logs operacionais).
3. Ingestão controlada do `RBAC43EMD05.pdf` (`scripts/ingest_regulation.py`), inspeção (`scripts/inspect_document.py`), revisão e aprovação humana explícita (`scripts/validate_extraction.py`) e indexação (`scripts/index_document.py`) com a cota Free Tier disponível.
4. Implementação completa da **Etapa 4 — Bot Telegram** na branch dedicada `feat/etapa-4-telegram-bot` com testes 100% offline (incluindo adaptador de auditoria em `/app/logs` e rotação de logs no Compose, quando autorizado).
5. **Etapa 5 — Operação contínua e calibração real:** calibração do limiar provisório `max_vector_distance=0.85` sobre o RBAC 43 real e rotina de backup/restore.
