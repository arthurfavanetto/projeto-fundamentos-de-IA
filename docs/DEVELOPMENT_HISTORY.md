# Histórico de Desenvolvimento e Continuidade — RBAC AI Assistant

Este documento registra a linha do tempo real do repositório `arthurfavanetto/projeto-fundamentos-de-IA` baseada exclusivamente nos commits, Pull Requests mergeados, ADRs e resultados de validação verificados, permitindo que qualquer nova sessão retome o desenvolvimento sem depender do histórico de chat.

---

## 1. Linha do Tempo de Pull Requests Integrados à `main`

| PR | Branch de origem | Merge Commit (`main`) | Data (UTC) | Marco / Entrega Principal |
| :---: | --- | --- | --- | --- |
| **#1–#3** | `arthurfavanetto-patch-1/2` | `02dc828` / `de81a01` | 2026-11-20 | Notas iniciais conceituais (posteriormente arquivadas em `docs/legacy/`). |
| **#4** | `docs/etapa-0-architecture-baseline` | `02d68ebcb36fd67b1da6d707b091159bb6afe1d6` | 2026-11-20 | **Etapa 0:** Arquitetura hexagonal (`src-layout`), ADRs 01–06, protocolo de fidelidade normativa, `.gitignore`, `.dockerignore`, `.env.example`, `config/settings.yaml`, `config/access_control.example.yaml`. |
| **#5** | `feat/etapa-1-foundation-docker` | `7e096ee53ee1fc4a000e13d6de83dfe05ef04be9` | 2026-11-20 | **Etapa 1:** Fundação Python 3.12 (`pyproject.toml`, `uv.lock`, `AppSettings`, `logging_config`, `secret_redactor`, `healthcheck.py`, `main.py`), `Dockerfile` multi-stage non-root (UID/GID `10001`), `docker/chromadb.Dockerfile` (`chromadb/chroma:1.5.9`), `docker-compose.yml` com isolamento de rede interna e CI GitHub Actions. |
| **#6** | `feat/etapa-2-rbac43-ingestion` | `5ebc919902ddc8fb86f744951a55646a92304655` | 2026-11-20 | **Etapa 2:** Motor multidocumento de ingestão regulatória — SQLite SSOT (`v001`), `pdfplumber`, dupla paginação (PDF + impressa), parser estrutural ANAC/genérico, política híbrida `review_on_demand`, bloqueio estrito de OCR não revisado e CLIs (`ingest_regulation.py`, `inspect_document.py`, `validate_extraction.py`). |
| **#7** | `feat/etapa-3-gemini-hybrid-rag` | `acb600d61416c3269fbedb1b8d9f1e84443ff789` | 2026-11-22 | **Etapa 3:** Integração oficial `google-genai` sob política única Free Tier (`gemini-4-hydrogen` + `gemini-embedding-2` 768d, `GEMINI_ALLOW_PAID_MODELS=false`, ADR-07/ADR-08), migração SQLite FTS5 (`v002`), `ChromaVectorStore` com filtro `amendment_number_key`, motor RAG híbrido (4 modos + RRF + reranker determinístico + poda adaptativa), `PromptGuard`, `CitationValidator` (5 dimensões), `NormativeResponseFormatter` dual, CLI `query_assistant.py` (`--debug`/`--json`/`--no-llm`), avaliação quantitativa com 40 consultas sintéticas (`scripts/evaluate_rag.py`) e CI atualizado para Node 24. |
| **#9** | `fix/windows-test-compatibility` | `4d4673bc19fecf661ce1a048acbe9abf5de7cd4b` | 2026-11-22 | **Portabilidade Windows, Isolamento e Performance:** `StorageConfig.canonical_container_sqlite_path` (`PurePosixPath`), logs CLI em `stderr` (`stdout` JSON puro), isolamento `.env` (`RBAC_DISABLE_DOTENV=1`) e bloqueio de chamadas Gemini reais (`RBAC_BLOCK_REAL_GEMINI_API=1`) em imports/testes/workers/subprocessos, SQLite `:memory:` em fixtures, dedup N+1 por requisição em `HybridNormativeRetriever`, fechamento explícito de `SqliteDocumentRepository` e `PersistentClient`, e `pytest-xdist` (`3.8.0`). Validado localmente no Windows 11 (`90 passed`, `91%` cov, `44.63s` seq / `38.82s` paralelo — corrigindo travamento anterior de 56 min). |
| **#8** | `perf/eval-query-embedding-cache` | `d1d6af150b0a20292ae7f91e33da0b91146573ca` | 2026-11-22 | **Otimização do Benchmark:** `QueryEmbeddingCacheKey` + `InMemoryQueryEmbeddingCache` em `scripts/evaluate_rag.py` exclusivamente para o caminho real de avaliação (`--embedding-backend compare`), reduzindo `1.286` → `46` chamadas HTTP simuladas (-96,42%, `6` documentais + `40` consultas únicas, `0` geração, métricas idênticas, limpeza em `finally`, zero persistência). CI Run `45068432151` aprovado (`93 passed`, `91%` cov). |

---

## 2. Decisões Arquiteturais Consolidadas (ADRs)

- **ADR-01 / ADR-02:** Clean/Hexagonal Architecture + separação estrita entre `TEXTO NORMATIVO` (literal do SQLite SSOT aprovado) e `EXPLICAÇÃO` (gerada por IA, validada pelo `CitationValidator`).
- **ADR-03:** Telegram via **HTTPS Long Polling** (sem exposição de portas ou webhooks — preparado em config/docs, pendente de implementação na Etapa 4).
- **ADR-04 / ADR-07:** Recuperação híbrida em 3 sinais (**Estrutural exato** + **SQLite FTS5 BM25** + **ChromaDB 1.5.9 cosseno**) fundidos por **RRF**, reranker determinístico, poda adaptativa e hidratação ao vivo no SQLite (limiar provisório `max_vector_distance=0.85`).
- **ADR-05:** Topologia mínima de 2 containers non-root (`rbac-bot` + `rbac-chromadb`, UID/GID `10001`, rede interna isolada `rbac_internal_net`) e 3 volumes nomeados (`chroma_data`, `regulatory_documents`, `audit_logs`).
- **ADR-06:** Segredos e IDs nunca versionados (`.env`, `config/access_control.yaml`, `data/`, `*.pdf` no `.gitignore` e `.dockerignore`).
- **Política única Gemini Free Tier:** `GEMINI_GENERATION_MODEL=gemini-4-hydrogen`, `GEMINI_EMBEDDING_MODEL=gemini-embedding-2`, `GEMINI_EMBEDDING_DIMENSIONS=768`, `GEMINI_ALLOW_PAID_MODELS=false`, sem fallback silencioso e sem Cloud Billing.

---

## 3. Evolução de Benchmarks e Qualidade (`d1d6af150b0a20292ae7f91e33da0b91146573ca`)

- **Avaliação sintética RAG (40 consultas, 19 categorias):** `hybrid_rrf` alcança `Rec@5 = 100.0%`, `Prec@5 = 85.7%`, `Prec@1 = 100.0%`, `MRR@10 = 1.000`, `nDCG@10 = 1.000`, `0` violações de governança e `0` citações alucinadas.
- **Testes e cobertura (VM Linux 2 vCPUs, Python 3.12.16):** `93 passed`, `0 failed`, `0 skipped`, `91%` cobertura (`4.741` stmts, `442` missed) idêntica nos modos sequencial (`48.41s`) e paralelo (`29.47s` com `-n auto --dist load`, `29.82s` com `-n auto --dist loadscope`); Ruff e MyPy strict (`82` arquivos) aprovados.
- **Validação anterior no Windows 11 (commit `838b58f` do PR #9):** `90 passed` em `44.63s` (sequencial) e `38.82s` (`pytest-xdist`), `91%` cobertura idêntica, corrigindo as 3 falhas originais e o travamento de 56 minutos.

---

## 4. Incidente de 22/11/2026 e Retomada Limpa

- **Incidente:** execução indevida de `docker compose down -v` no host Windows removeu os 3 volumes nomeados do projeto (não havia documentos oficiais RBAC ingeridos nem aprovações humanas registradas).
- **Decisão definitiva (23/11/2026):** recuperação forense dos arquivos VHDX encerrada; backups VHDX preservados fisicamente fora do fluxo; reconstrução limpa usando Docker Compose com volumes novos/vazios a partir da `main` (`d1d6af1`).
- **Documento RBAC 43 (`RBAC43EMD05.pdf`, 32 páginas):** SHA-256 oficial `5026d2da87fc3f55c61a6b5a20dd1f50d52a1367a8550897f3864074e1a46472` cadastrado em `config/regulations_catalog.yaml`; ingestão, revisão humana explícita e indexação aguardam a recriação dos volumes no Windows.
- **Telegram (Etapa 4):** ainda não implementado (placeholder em `src/rbac_assistant/interfaces/telegram/__init__.py`); próximo ciclo de desenvolvimento após validação local do Docker e do RAG no Windows.

---

## 5. Próximas Etapas do Roadmap

1. **Reconstrução local no Windows 11:** liberar `/dev/sdd` no WSL (se ainda anexado), subir Docker Desktop, recriar os 3 volumes vazios, validar healthchecks, persistência, logs e suíte de testes no Windows.
2. **Ingestão e validação do RBAC 43:** ingerir `RBAC43EMD05.pdf`, inspecionar, revisar/aprovar páginas manualmente (`scripts/validate_extraction.py`) e indexar no SQLite FTS5 + ChromaDB (`scripts/index_document.py`) respeitando o Free Tier.
3. **Etapa 4 (`feat/etapa-4-telegram-bot`):** biblioteca Telegram, Long Polling, controle de acesso `deny_by_default`, `MessageSplitter`, handlers integrados ao `NormativeChatService`, adaptador de auditoria em `/app/logs`, rotação de logs no Compose e testes 100% offline.
4. **Etapa 5:** calibração real de `max_vector_distance` sobre o RBAC 43 e rotina contínua de backup/restore.
