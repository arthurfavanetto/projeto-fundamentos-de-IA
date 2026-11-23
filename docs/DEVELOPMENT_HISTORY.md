# Histórico de Desenvolvimento e Continuidade — RBAC AI Assistant

Este documento registra a linha do tempo real do repositório `arthurfavanetto/projeto-fundamentos-de-IA` baseada exclusivamente nos commits, Pull Requests integrados, ADRs e resultados de validação verificados.

---

## 1. Linha do Tempo de Pull Requests Integrados à `main`

| PR | Branch de origem (preservada) | Merge Commit (`main`) | Data (UTC) | Marco / Entrega Principal |
| :---: | --- | --- | --- | --- |
| **#1–#3** | `arthurfavanetto-patch-1/2` | `02dc828` / `de81a01` | 2026-11-20 | Notas iniciais conceituais (arquivadas em `docs/legacy/`). |
| **#4** | `docs/etapa-0-architecture-baseline` | `02d68ebcb36fd67b1da6d707b091159bb6afe1d6` | 2026-11-20 | **Etapa 0:** Arquitetura hexagonal (`src-layout`), ADRs 01–06, protocolo de fidelidade, `.gitignore`, `.dockerignore` e templates. |
| **#5** | `feat/etapa-1-foundation-docker` | `7e096ee53ee1fc4a000e13d6de83dfe05ef04be9` | 2026-11-20 | **Etapa 1:** Fundação Python 3.12 (`pyproject.toml`, `uv.lock`, `AppSettings`, `logging_config`, `secret_redactor`, `healthcheck.py`, `main.py`), `Dockerfile` non-root (`10001`), `chromadb/chroma:1.5.9`, `docker-compose.yml` e CI. |
| **#6** | `feat/etapa-2-rbac43-ingestion` | `5ebc919902ddc8fb86f744951a55646a92304655` | 2026-11-20 | **Etapa 2:** Ingestão regulatória multidocumento — SQLite SSOT (`v001`), `pdfplumber`, dupla paginação, parser ANAC/genérico, política `review_on_demand` e CLIs. |
| **#7** | `feat/etapa-3-gemini-hybrid-rag` | `acb600d61416c3269fbedb1b8d9f1e84443ff789` | 2026-11-22 | **Etapa 3:** `google-genai` Free Tier (`gemini-4-hydrogen` + `gemini-embedding-2` 768d, ADR-07/08), SQLite FTS5 (`v002`), `ChromaVectorStore`, RAG híbrido 4 modos + RRF + reranker, `PromptGuard`, `CitationValidator`, `NormativeResponseFormatter` e `scripts/evaluate_rag.py`. |
| **#9** | `fix/windows-test-compatibility` | `4d4673bc19fecf661ce1a048acbe9abf5de7cd4b` | 2026-11-22 | **Portabilidade Windows & Performance:** `PurePosixPath`, CLI `--json` puro (`stderr`), isolamento `.env`, bloqueio Gemini real em testes, SQLite `:memory:`, dedup N+1 e `pytest-xdist` (validado no Windows 11 em `838b58f`). |
| **#8** | `perf/eval-query-embedding-cache` | `d1d6af150b0a20292ae7f91e33da0b91146573ca` | 2026-11-22 | **Cache de Benchmark:** `InMemoryQueryEmbeddingCache` em `scripts/evaluate_rag.py` (`1.286` → `46` chamadas simuladas no modo `compare`, CI `45068432151`). |
| **#10** | `chore/docker-log-rotation` | `292c11667f7fef145add130a7ff07eec9747a2bf` | 2026-11-23 | **Rotação de Logs Docker:** `json-file` (`max-size: "10m"`, `max-file: "5"`) nos serviços `chromadb` e `rbac-bot` + testes de contrato (CI `45270448546`). |
| **#11** | `chore/docker-runtime-recovery` | `5112bd5048175a932e746b21225424eeed39568f` | 2026-11-23 | **Documentação de Recuperação:** 5 documentos de estado, recuperação pós-incidente, setup Docker, validação e histórico (CI `45271482437`). |
| **#12** | `feat/persistent-audit-adapter` | `571b0b67c3ef737972ed4008bedf66a0aa6c8f2b` | 2026-11-23 | **Auditoria Persistente (`AuditPort`):** `JsonlAuditAdapter` em `/app/logs/audit_events.jsonl` (UTC ISO-8601, redação obrigatória + fix regex Telegram, concorrência thread/processo, rotação por tamanho, `AuditWriteError`), integração hexagonal, `scripts/verify_audit_persistence.py` no CI Docker e `docs/operations/AUDIT_LOGGING.md` (CI `45277532713`). |
| **#13** | `feat/etapa-4-telegram-bot` | `65dc60f9e46e29ca83e8fa632d5f691b2fb66014` | 2026-11-23 | **Etapa 4 — Bot Telegram:** `python-telegram-bot==22.8`, Long Polling + shutdown gracioso, `AccessControl` `deny_by_default` fail-closed, `UserRateLimiter`, `MessageSplitter` UTF-16 fiel, 8 comandos (`/start`, `/help`, `/status`, `/rbac`, `/buscar`, `/explicar`, `/documentos`, `/versao`) + texto livre, thread-safety SQLite, guarda `RBAC_BLOCK_REAL_TELEGRAM_API=1`, mount `./config:/app/config:ro` no Compose e `docs/operations/TELEGRAM_SETUP.md` (CI `45279737245`). |
| **#14** | `feat/etapa-5-rag-evaluation-quality` | `ff74031ee3eabc6eaa37175308aec2b5a461b4ca` | 2026-11-23 | **Etapa 5 — Avaliação Oficial & Regressão Normativa:** dataset oficial RBAC 43 Emenda 05 (`config/evaluations/rbac43_emd05_eval_dataset.json`), `corpus_evaluator.py` com gate de aprovação humana e recomendação reproduzível de calibração (mantendo `max_vector_distance=0.85` provisório), `--corpus official-approved` em `scripts/evaluate_rag.py`, `test_regulatory_fidelity_regression.py` e `docs/operations/RAG_EVALUATION_AND_INGESTION.md` (CI `45280707108`). |
| **#15** | `feat/backup-restore-and-final-docs` | *(PR em integração)* | 2026-11-23 | **Backup/Restauração Verificável & Consolidação Documental:** `backup_manager.py`, `scripts/backup_restore.py`, `scripts/backup_docker_volumes.ps1`, `scripts/restore_docker_volumes.ps1`, `test_backup_restore.py`, `docs/operations/BACKUP_AND_RESTORE.md` e atualização integral dos 5 documentos canônicos. |

---

## 2. Decisões Arquiteturais Consolidadas (ADRs)

- **ADR-01 / ADR-02:** Clean/Hexagonal Architecture + separação estrita entre `TEXTO NORMATIVO` (literal do SQLite SSOT aprovado) e `EXPLICAÇÃO` (gerada por IA, validada pelo `CitationValidator`).
- **ADR-03:** Telegram via **HTTPS Long Polling** (`python-telegram-bot==22.8`, sem webhooks nem exposição de portas).
- **ADR-04 / ADR-07:** Recuperação híbrida em 3 sinais (**Estrutural exato** + **SQLite FTS5 BM25** + **ChromaDB 1.5.9 cosseno**) fundidos por **RRF**, reranker determinístico, poda adaptativa e hidratação ao vivo no SQLite (`max_vector_distance=0.85` mantido provisório até calibração real).
- **ADR-05:** Topologia mínima de 2 containers non-root (`rbac-bot` + `rbac-chromadb`, UID/GID `10001`, rede interna `rbac_internal_net`), 3 volumes nomeados (`chroma_data`, `regulatory_documents`, `audit_logs`), montagem `./config:/app/config:ro` e rotação `json-file` (`10m` x `5`).
- **ADR-06:** Segredos e IDs nunca versionados (`.env`, `config/access_control.yaml`, `data/`, `*.pdf`, `*.jsonl` no `.gitignore`).
- **ADR-08:** Política única Gemini Free Tier (`GEMINI_GENERATION_MODEL=gemini-4-hydrogen`, `GEMINI_EMBEDDING_MODEL=gemini-embedding-2`, `GEMINI_EMBEDDING_DIMENSIONS=768`, `GEMINI_ALLOW_PAID_MODELS=false`, sem fallback silencioso e sem Cloud Billing).

---

## 3. Incidente de 22/11/2026 e Retomada Limpa

- **Incidente:** execução indevida de `docker compose down -v` no host Windows removeu os 3 volumes antigos do projeto (que não possuíam documentos oficiais ingeridos nem aprovações humanas registradas).
- **Decisão definitiva (23/11/2026):** recuperação forense VHDX encerrada; backups VHDX preservados intactos fora do fluxo; reconstrução limpa no Windows 11 com volumes novos e vazios, acompanhada agora de scripts verificáveis de backup e restauração com SHA-256 (`scripts/backup_docker_volumes.ps1` e `scripts/restore_docker_volumes.ps1`).
