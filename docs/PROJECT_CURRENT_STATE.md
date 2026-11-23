# Estado Atual do Projeto — RBAC AI Assistant

> **Referência de código:** Etapas 0–5 integradas via PRs #4 a #16 + modo público do Bot Telegram (`feat/telegram-public-access-mode`).
>
> **Domínio:** Regulamentos Brasileiros da Aviação Civil (**RBAC — ANAC**; nunca *Role-Based Access Control*).
>
> **Regra de veracidade:** este documento distingue rigorosamente o que está **implementado e testado em CI/VM Linux**, o que já foi **validado em Python no Windows 11** e o que permanece **PENDENTE DE VALIDAÇÃO LOCAL NO WINDOWS 11** ou **dependente de ação humana**.

---

## 1. Arquitetura Consolidada (Hexagonal / Clean Architecture)

```text
src/rbac_assistant/
├── config/           # AppSettings (Pydantic Settings + YAML + RBAC_DISABLE_DOTENV), logging_config (stdout/stderr)
├── domain/           # Entidades normativas, exceções (incl. AuditWriteError) e ports (Protocol) — zero dependências externas
├── application/      # CatalogService, IngestionService, ValidationService, IndexingService, NormativeChatService e use cases
├── infrastructure/
│   ├── audit/        # JsonlAuditAdapter (AuditPort -> /app/logs/audit_events.jsonl, UTC ISO-8601, pseudonimização, redação, concorrência, rotação)
│   ├── llm/          # GeminiClientAdapter + GeminiEmbeddingAdapter (google-genai==3.6.1, política única Free Tier)
│   ├── pdf/          # PdfPlumberExtractorAdapter, AnacRbacStructureStrategy, GenericStructureStrategy, TextQualityChecker, TesseractOcrAdapter
│   ├── sources/      # LocalFileSourceAdapter, AnacHttpSourceAdapter
│   ├── storage/      # SqliteDocumentRepository (SSOT ACID + WAL + FTS5 + thread-safe), LocalFileDocumentStore, backup_manager (SHA-256)
│   └── vector_db/    # ChromaVectorStore (ChromaDB 1.5.9, guarda de compatibilidade modelo/dimensão/estratégia)
├── rag/
│   ├── ingestion/    # StructuralNormativeChunker
│   ├── retrieval/    # SectionResolver, HybridNormativeRetriever (Estrutural + FTS5 + ChromaDB → RRF + Reranker + Poda; escopo aprovado no Telegram)
│   ├── generation/   # PromptTemplateStore, CitationValidator, NormativeResponseFormatter
│   └── evaluation/   # RagEvaluator (40 consultas sintéticas) + corpus_evaluator (avaliação oficial RBAC 43 + recomendação de calibração)
├── security/         # AccessControl (modo público USER + 2 ADMINs isolados / public_access=false), UserRateLimiter (usuário + rajada + global), PromptGuard, secret_redactor
├── interfaces/
│   └── telegram/     # TelegramCommandRouter (9 comandos incl. /id + texto livre, /status público sanitizado vs ADMIN, timeout + semáforo global), MessageSplitter (4.096 UTF-16), TelegramBotRunner (Long Polling, PTB_TIMEDELTA=1, somente conversas privadas)
├── healthcheck.py    # CLI de healthcheck Docker (ChromaDB heartbeat + escrita nos volumes)
└── main.py           # Composition Root completo (auditoria JSONL, SQLite, ChromaDB, RAG e Bot Telegram com modo standby seguro)
```

---

## 2. Matriz Completa de Funcionalidades por Ambiente de Validação

| Funcionalidade / Módulo | Código | VM Linux (Offline) | GitHub Actions CI | Windows 11 Local | Observação / Pendência Real |
| --- | :---: | :---: | :---: | :---: | --- |
| Arquitetura Hexagonal + ADRs 01–08 | **Implementado** | **Testado** | **Testado** | Validado em Python (`c607a7f`) | Revalidar no Windows após `git pull` |
| SQLite SSOT + WAL + FTS5 + Thread-Safety | **Implementado** | **Testado** | **Testado** (unit + Docker) | **Pendente** no Docker Desktop | Recriar volume `regulatory_documents` |
| Ingestão `pdfplumber` + Revisão Humana (`review_on_demand`) | **Implementado** | **Testado** | **Testado** | **Pendente** (ação humana) | Ingerir `RBAC43EMD05.pdf` e revisar páginas manualmente |
| ChromaDB 1.5.9 + Política Única Gemini Free Tier (`gemini-4-hydrogen` / `gemini-embedding-2` 768d) | **Implementado** | **Testado** (mocks) | **Testado** (mocks + Docker Chroma) | **Pendente** | Indexação real e consultas Gemini dependem da cota Free Tier local |
| RAG Híbrido + Escopo Normativo Estrito (somente páginas aprovadas no Telegram, `PromptGuard`, `CitationValidator`) | **Implementado** | **Testado** | **Testado** | **Pendente** | Validar no corpus oficial aprovado do RBAC 43 |
| Rotação de Logs Operacionais Docker (`json-file`, `10m` x `5`) | **Implementado** (PR #10) | Contrato testado | **Testado em Docker real** | **Pendente** | Validar via `docker inspect` no Docker Desktop Windows |
| Auditoria Persistente JSONL (`JsonlAuditAdapter` em `/app/logs`, `pseudonymize_user_ids: true`) | **Implementado** (PR #12) | **Testado** | **Testado em Docker real pós-restart** | **Pendente** | Validar com `scripts/verify_audit_persistence.py` no Windows |
| Compatibilidade `PTB_TIMEDELTA=1` + Normalização `RetryAfter` (`0 warnings`) | **Implementado** (PR #16) | **Testado** (`0 warnings`) | **Testado** (pytest + container `rbac-bot`) | **Pendente** de revalidação pós-pull | Confirmar `0 warnings` no Windows após `git pull` |
| Bot Telegram Público (conversas privadas, `public_access: true`, `USER` padrão, 2 `ADMIN`s isolados, `/id`, `/status` sanitizado, rate limit global + concorrência global + timeout) | **Implementado** (`feat/telegram-public-access-mode`) | **Testado** (mocks) | **Testado** (mocks + Docker standby) | **Pendente** (ação humana) | Obter os dois IDs via `/id`, configurar `config/access_control.yaml` local e testar no Telegram real |
| Avaliação Oficial RBAC 43 + Calibração Reproduzível (`max_vector_distance=0.85` provisório) + Regressão Normativa | **Implementado** (PR #14) | **Testado** | **Testado** | **Pendente** (ação humana) | Executar `--corpus official-approved --embedding-backend gemini-embedding-2` após aprovação humana no Windows |
| Backup & Restauração Verificável (UTC, SHA-256, SQLite `integrity_check`, scripts `.ps1`) | **Implementado** (PR #15) | **Testado** | **Testado** | **Pendente** | Executar `scripts/backup_docker_volumes.ps1` e drill de restauração no Windows |

---

## 3. Documentação Operacional Específica

- **Docker, volumes e logs:** [`docs/DOCKER_SETUP.md`](./DOCKER_SETUP.md)
- **Auditoria persistente JSONL:** [`docs/operations/AUDIT_LOGGING.md`](./operations/AUDIT_LOGGING.md)
- **Bot Telegram público, 2 administradores e `/id`:** [`docs/operations/TELEGRAM_SETUP.md`](./operations/TELEGRAM_SETUP.md)
- **Ingestão do RBAC 43, revisão humana, escopo estrito e avaliação/calibração RAG:** [`docs/operations/RAG_EVALUATION_AND_INGESTION.md`](./operations/RAG_EVALUATION_AND_INGESTION.md)
- **Backup, verificação SHA-256 e restauração:** [`docs/operations/BACKUP_AND_RESTORE.md`](./operations/BACKUP_AND_RESTORE.md)
- **Recuperação pós-incidente de 22/11/2026:** [`docs/RECOVERY_STATUS.md`](./RECOVERY_STATUS.md)

---

## 4. Critérios Ainda Pendentes para Declarar Operação 100% Concluída no Windows 11

O código-fonte, a arquitetura, a suíte automatizada e os pipelines de CI estão atualizados no GitHub. Para declarar o sistema **100% operacional em produção local**, ainda dependem exclusivamente de execução no computador Windows 11 do proprietário:

1. Atualizar o clone local (`git pull --ff-only origin main`) e revalidar `ruff`, `mypy` e `pytest` (`0 warnings`).
2. Verificação/liberação específica de `F:\DockerRecovery\docker_data_working.vhdx` no WSL (somente se ainda anexado).
3. Inicialização do Docker Desktop Windows e recriação limpa dos 3 volumes (`chroma_data`, `regulatory_documents`, `audit_logs`) via `docker compose up -d --build`.
4. Validação local de healthchecks, persistência (ChromaDB, SQLite, auditoria JSONL), `PTB_TIMEDELTA=1`, rotação `json-file` e presença das credenciais no `.env` (sem exposição).
5. Descoberta dos Telegram User IDs dos dois administradores via `/id` em conversa privada, criação de `config/access_control.yaml` local com `public_access: true` e `roles.admins` preenchido, e teste real do Bot Telegram (usuário público + os 2 administradores + recusa fora do escopo).
6. Ingestão do `RBAC43EMD05.pdf`, revisão e aprovação humana explícita das páginas, indexação Free Tier, avaliação real (`--corpus official-approved`) e primeiro backup verificado (`scripts/backup_docker_volumes.ps1`).
