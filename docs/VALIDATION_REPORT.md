# Relatório de Validação Técnica — Reconstrução (22/11/2026)

> **Ambiente de validação desta rodada:** VM Linux Debian 12 do Antigravity (2 vCPUs `Intel Xeon Platinum 8481C`, 7.8 GiB RAM, `Python 3.12.16` via `uv 0.13.7`), sobre o commit consolidado `main` = `d1d6af150b0a20292ae7f91e33da0b91146573ca`.  
> **Isolamento:** todas as execuções ocorreram com `env -u GEMINI_API_KEY`, `RBAC_DISABLE_DOTENV=1` e `RBAC_BLOCK_REAL_GEMINI_API=1` (0 chamadas reais à API Gemini, 0 chamadas à API Telegram).

---

## 1. Versões e Referências Git Verificadas

- **Repositório:** `arthurfavanetto/projeto-fundamentos-de-IA`
- **Branch `main` / `origin/main`:** `d1d6af150b0a20292ae7f91e33da0b91146573ca`
- **Merge Commit PR #9:** `4d4673bc19fecf661ce1a048acbe9abf5de7cd4b`
- **Merge Commit PR #8:** `d1d6af150b0a20292ae7f91e33da0b91146573ca`
- **Python do projeto (`.venv`):** `3.12.16`
- **`uv`:** `0.13.7` (`uv sync --frozen --check`: `115` pacotes consistentes com `uv.lock`, zero alterações)
- **Pytest / Plugins:** `pytest 9.2.0`, `pytest-cov 7.1.0`, `pytest-asyncio 1.5.0`, `pytest-xdist 3.8.0` (`execnet 2.1.2`)
- **PDF local de referência (`data/raw_pdfs/RBAC43EMD05.pdf`, gitignored):** presente (`349.196` bytes), SHA-256 `5026d2da87fc3f55c61a6b5a20dd1f50d52a1367a8550897f3864074e1a46472` (compatível com `config/regulations_catalog.yaml`).

---

## 2. Resultados Reais — Lint, Tipagem, Unitários, Integração e CLI Offline

| Etapa | Comando | Resultado | Aprovados | Falhas | Ignorados | Duração |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Lint (`Ruff`) | `env -u GEMINI_API_KEY uv run ruff check src tests scripts` | **PASS** | — | 0 | — | < 1s |
| Tipagem (`MyPy strict`) | `env -u GEMINI_API_KEY uv run mypy src` | **PASS** (`82` arquivos) | — | 0 | — | ~1s |
| Etapa A — Unitários | `env -u GEMINI_API_KEY uv run pytest tests/unit -v` | **PASS** | 75 | 0 | 0 | `10.42s` |
| Etapa B — Integração + Evals | `env -u GEMINI_API_KEY uv run pytest tests/integration tests/evals -v` | **PASS** | 18 | 0 | 0 | `12.21s` |
| CLI offline (`query_assistant.py --help`) | `uv run python scripts/query_assistant.py --help` | **PASS** | — | 0 | — | < 1s |
| CLI offline (base vazia: `inspect_document.py`, `query_assistant.py --no-llm --json/--debug`) | Execução em `TemporaryDirectory` com `RBAC_DISABLE_DOTENV=1` | **PASS** (`[]`, JSON puro em `stdout`, `stderr` vazio, recusa segura `found=False`, bloco `--debug` íntegro) | — | 0 | — | ~2s |

---

## 3. Resultados Reais — Suíte Completa Sequencial e Benchmark Paralelo (14 Cenários `pytest-xdist`)

| Cenário | Modo / Flags | Wall time | Tempo pytest | Pico RAM (árvore) | CPU sist. | Cobertura (`pytest-cov`) | Resultado |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1. Direcionados (4 arq.) | Sequencial | `5.93s` | `4.87s` | `261.8 MB` | `43.0%` | n/a | `51 passed, 0 failed` |
| 2. Direcionados (4 arq.) | `-n 4 --dist load` | `7.40s` | `6.47s` | `766.8 MB` | `86.0%` | n/a | `51 passed, 0 failed` |
| 3. Direcionados (4 arq.) | `-n 8 --dist load` | `12.25s` | `11.31s` | `1278.1 MB` | `90.5%` | n/a | `51 passed, 0 failed` |
| 4. Direcionados (4 arq.) | `-n auto --dist load` (2w) | `5.53s` | `4.48s` | `491.4 MB` | `75.7%` | n/a | `51 passed, 0 failed` |
| 5. Completa sem `--cov` | Sequencial | `24.12s` | `22.91s` | `450.7 MB` | `48.4%` | n/a | `93 passed, 0 failed` |
| 6. Completa sem `--cov` | `-n 4 --dist load` | `16.63s` | `15.70s` | `894.2 MB` | `92.2%` | n/a | `93 passed, 0 failed` |
| 7. Completa sem `--cov` | `-n 8 --dist load` | `23.66s` | `22.68s` | `1517.1 MB` | `89.3%` | n/a | `93 passed, 0 failed` |
| 8. Completa sem `--cov` | `-n auto --dist load` (2w) | `15.55s` | `14.48s` | `685.6 MB` | `85.0%` | n/a | `93 passed, 0 failed` |
| 9. Completa sem `--cov` | `-n auto --dist loadscope` (2w) | **`14.49s`** | **`13.54s`** | `607.5 MB` | `88.4%` | n/a | `93 passed, 0 failed` |
| 10. Completa **com `--cov`** | Sequencial | `50.75s` | `48.41s` | `451.9 MB` | `49.9%` | **`91%` (`4741` / `442`)** | `93 passed, 0 failed` |
| 11. Completa **com `--cov`** | `-n 4 --dist load` | `36.95s` | `34.73s` | `920.9 MB` | `94.2%` | **`91%` (`4741` / `442`)** | `93 passed, 0 failed` |
| 12. Completa **com `--cov`** | `-n 8 --dist load` | `51.82s` | `49.75s` | `1481.3 MB` | `92.7%` | **`91%` (`4741` / `442`)** | `93 passed, 0 failed` |
| 13. Completa **com `--cov`** | `-n auto --dist load` (2w) | **`31.62s`** | **`29.47s`** | `701.4 MB` | `90.6%` | **`91%` (`4741` / `442`)** | `93 passed, 0 failed` |
| 14. Completa **com `--cov`** | `-n auto --dist loadscope` (2w) | `32.07s` | `29.82s` | `629.3 MB` | `90.0%` | **`91%` (`4741` / `442`)** | `93 passed, 0 failed` |

### Consistência de cobertura e módulos sem cobertura (0%)
- **Cobertura global:** **`91%`** (`4.741` statements, `442` missed), **100% idêntica** entre sequencial e todos os cenários paralelos (`-n 4`, `-n 8`, `-n auto load`, `-n auto loadscope`).
- **Arquivos com `0%` de cobertura** (contratos/casos de uso ainda não conectados em runtime):
  - `src/rbac_assistant/application/use_cases/register_document.py` (`9` stmts)
  - `src/rbac_assistant/domain/ports/audit_port.py` (`15` stmts)
  - `src/rbac_assistant/domain/ports/document_store_port.py` (`10` stmts)
- **Módulos com menor cobertura parcial:** `ocr_adapter.py` (`49%`), `main.py` (`68%`), `chroma_vector_store.py` (`72%`), `file_document_store.py` (`79%`).

---

## 4. Validação Docker (CI GitHub Actions vs. Host Windows Local)

| Verificação Docker | GitHub Actions CI (`45068432151` em `d1d6af1`) | Host Windows 11 local (pós-incidente) |
| --- | --- | --- |
| `docker compose config --quiet` | **PASS** | **Pendente** (aguarda liberação `/dev/sdd` e abertura do Docker Desktop) |
| Build + subida (`rbac-chromadb` + `rbac-bot`) | **PASS** | **Pendente** |
| Healthchecks (`GET /api/v2/heartbeat` + `healthcheck.py`) | **PASS** | **Pendente** |
| Execução non-root UID/GID `10001` e isolamento de rede | **PASS** | **Pendente** |
| Persistência ChromaDB após restart | **PASS** | **Pendente** |
| Persistência SQLite, PDF original e revisão após restart | **PASS** | **Pendente** |

---

## 5. Avaliação Quantitativa Offline do Motor RAG (`scripts/evaluate_rag.py`)

Executado com `env -u GEMINI_API_KEY uv run python scripts/evaluate_rag.py` sobre o corpus sintético controlado (`40` consultas, embedder determinístico `768d`, **0 chamadas reais à Gemini API**):

| Modo (`max_vector_distance=0.85` provisório) | Rec@5 | Prec@5 | Prec@1 | MRR@10 | nDCG@10 | Estrut | Granul | DocFlt | VerFlt | Abst | GovV | HalCit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `structural_only` | `40.6%` | `40.6%` | `40.6%` | `0.406` | `0.406` | `100.0%` | `39.4%` | `100.0%` | `100.0%` | `100.0%` | `0` | `0` |
| `lexical_fts5_only` | `100.0%` | `71.4%` | `90.6%` | `0.948` | `0.961` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `0` | `0` |
| `vector_only` | `81.2%` | `41.7%` | `71.9%` | `0.766` | `0.778` | `30.8%` | `81.8%` | `100.0%` | `100.0%` | `100.0%` | `0` | `0` |
| **`hybrid_rrf`** | **`100.0%`** | **`85.7%`** | **`100.0%`** | **`1.000`** | **`1.000`** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`0`** | **`0`** |

- **Auditoria do `CitationValidator` (mock determinístico):** `5/5` sondas (`100.0%` em IDs `[EV-N]`, referências estruturais, páginas PDF/impressa, bloqueio de citação inexistente e bloqueio de seção errada).
- **Ablação:** `baseline_rrf_k60_top5` (`Prec@5 67.8%`, `RetFP 39`) → `hybrid_default_rerank_cutoff` (`Prec@5 85.7%`, `RetFP 11`, `MRR@10 1.000`).
- **Análise de sensibilidade (`max_vector_distance`, mantido em `0.85` PROVISÓRIO):**
  - `0.50`: `HybR@5 98.4%`, `HybP@5 96.9%`, `12/13` categorias, `Abst 100.0%`
  - `0.75`: `HybR@5 100.0%`, `HybP@5 90.1%`, `13/13` categorias, `Abst 100.0%`
  - `0.85` (padrão provisório): `HybR@5 100.0%`, `HybP@5 85.7%`, `13/13` categorias, `Abst 100.0%`
  - `0.95`: `HybR@5 100.0%`, `HybP@5 83.1%`, `13/13` categorias, mas **`Abst` cai para `66.7%`**
  - *Nota:* não calibrado para produção até avaliação real com `gemini-embedding-2` sobre o RBAC 43 oficial.
- **Cache de embeddings (`tests/unit/test_eval_embedding_cache.py`):** confirmado `46` chamadas simuladas (`6` documentais + `40` consultas únicas) vs `1.286` sem cache (**-96,42%**), `0` chamadas `generate_content`, métricas idênticas, cache reconstruído em memória por execução sem depender dos volumes antigos.

---

## 6. Pendências Abertas para Conclusão no Host Windows
1. Confirmar no WSL se `/dev/sdd` foi desmontado com segurança (decisão definitiva de não usar mais os VHDX forenses) e abrir o Docker Desktop no Windows 11.
2. Confirmar presença local de `.env` (sem exibir segredos) e do PDF `RBAC43EMD05.pdf` (SHA-256 exato).
3. Recriar os 3 volumes novos/vazios e subir `rbac-chromadb` + `rbac-bot` com `docker compose up -d --build`.
4. Validar healthchecks, presença de `TELEGRAM_BOT_TOKEN` no contêiner (sem revelar o valor), logs operacionais e testes de persistência (`verify_chroma_persistence.py` e `verify_sqlite_persistence.py`).
5. Reexecutar Ruff, MyPy e `pytest --cov=rbac_assistant` no clone Windows `C:\Projetos\projeto-fundamentos-de-IA` sobre `d1d6af1`.
6. Prosseguir com a ingestão controlada do `RBAC43EMD05.pdf` (após revisão e aprovação humana explícita) e, na sequência, a **Etapa 4 (Bot Telegram)** em branch dedicada `feat/etapa-4-telegram-bot`.
