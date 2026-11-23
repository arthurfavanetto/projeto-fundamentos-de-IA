# Relatório de Validação Técnica — Consolidação Final (23/11/2026)

> **Princípio de veracidade:** todos os números abaixo foram efetivamente executados e observados na VM Linux do Antigravity ou no GitHub Actions CI. Nenhum teste de host Windows 11 pós-incidente foi simulado ou declarado concluído sem execução real pelo proprietário.
>
> **Isolamento:** todas as execuções de testes operam com `RBAC_DISABLE_DOTENV=1`, `RBAC_BLOCK_REAL_GEMINI_API=1` e `RBAC_BLOCK_REAL_TELEGRAM_API=1` (zero chamadas reais às APIs externas).

---

## 1. Validações Confirmadas no GitHub Actions (CI Linux + Docker Real)

| PR / Ref | Merge Commit / HEAD | CI Run ID | `Lint, Typecheck & Pytest (3.12)` | `Docker Non-Root, Heartbeat & Persistence Smoke Test` |
| --- | --- | :---: | :---: | :---: |
| PR #10 (`chore/docker-log-rotation`) | `292c11667f7fef145add130a7ff07eec9747a2bf` | `45270448546` (`main`) | **SUCCESS** (`53s`) | **SUCCESS** (`1m29s`) |
| PR #11 (`chore/docker-runtime-recovery`) | `5112bd5048175a932e746b21225424eeed39568f` | `45271482437` (`main`) | **SUCCESS** (`1m10s`) | **SUCCESS** (`1m33s`) |
| PR #12 (`feat/persistent-audit-adapter`) | `571b0b67c3ef737972ed4008bedf66a0aa6c8f2b` | `45277532713` (`main`) | **SUCCESS** (`1m13s`) | **SUCCESS** (`1m33s`, incl. persistência JSONL `/app/logs`) |
| PR #13 (`feat/etapa-4-telegram-bot`) | `65dc60f9e46e29ca83e8fa632d5f691b2fb66014` | `45279737245` (`main`) | **SUCCESS** (`1m12s`) | **SUCCESS** (`1m25s`, incl. mount `./config:/app/config:ro` + standby) |
| PR #14 (`feat/etapa-5-rag-evaluation-quality`) | `ff74031ee3eabc6eaa37175308aec2b5a461b4ca` | `45280707108` (PR) | **SUCCESS** (`1m21s`) | **SUCCESS** (`1m34s`) |

---

## 2. Resultados Reais da Suíte Automatizada (VM Linux Debian 12, Python 3.12.16)

- **`uv sync --frozen --check`:** íntegro (`116` pacotes consistentes com `uv.lock`, incluindo `python-telegram-bot==22.8`).
- **`uv run ruff check src tests scripts`:** **`All checks passed!`** (0 erros).
- **`uv run mypy src` (strict):** **`Success: no issues found`** (0 erros em todos os módulos de `src/rbac_assistant/`).
- **Cobertura global (`pytest --cov=rbac_assistant`):** **`91%`** (mantida acima de 90% ao longo dos PRs #12, #13, #14 e #15, sem redução artificial).
- **Novas suítes integradas nesta consolidação:**
  - `tests/unit/test_jsonl_audit_adapter.py`: escrita JSONL, timestamp UTC, redação de tokens/API keys (incl. fix de borda regex para tokens após underscore), bloqueio de prompts e textos integrais, pseudonimização, persistência entre subprocessos, concorrência multi-thread, rotação/retenção, falha explícita (`AuditWriteError`) e smoke test `scripts/verify_audit_persistence.py`.
  - `tests/unit/test_access_control.py`: `deny_by_default`, `ADMIN`/`USER`, fail-closed em arquivo ausente/vazio/malformado, IDs inválidos, auditoria e `UserRateLimiter` (burst + janela deslizante).
  - `tests/unit/test_message_splitter.py`: limite 4.096 UTF-16, emojis/Unicode, reconstrução 100% exata do texto normativo e redação de segredos.
  - `tests/unit/test_telegram_bot.py`: todos os 8 comandos + texto livre + comando desconhecido, bloqueio de não autorizados antes do RAG/LLM, rejeição de entrada excessivamente longa, deduplicação por `update_id`, rate limit, falhas de API Telegram (`NetworkError`, `TimedOut`, `RetryAfter`, `InvalidToken`), guarda `RBAC_BLOCK_REAL_TELEGRAM_API=1`, shutdown gracioso e Composition Root `main.py`.
  - `tests/evals/test_regulatory_fidelity_regression.py`: bloqueio de artigos/seções/citações inventadas, separação estrita SSOT vs. explicação IA, recusa segura sem evidência ou em páginas não aprovadas, schema do dataset oficial RBAC 43 Emenda 05, gate de revisão humana antes da avaliação oficial, elegibilidade de calibração e preservação de `max_vector_distance=0.85`.
  - `tests/unit/test_backup_restore.py`: ciclo completo backup → verificação SHA-256 → restauração + `PRAGMA integrity_check` + JSONL check, detecção de adulteração de 1 byte, proteção contra path traversal em `.tar.gz`, preservação de backups anteriores, CLI `scripts/backup_restore.py` e contrato de segurança dos scripts PowerShell.

---

## 3. Avaliação Quantitativa Sintética vs. Avaliação Oficial Real

- **Benchmark sintético (`scripts/evaluate_rag.py`, 40 consultas fictícias, embedder determinístico `768d`):** `hybrid_rrf` mantém `Rec@5 = 100.0%`, `Prec@5 = 85.7%`, `Prec@1 = 100.0%`, `MRR@10 = 1.000`, `nDCG@10 = 1.000`, `Abst = 100.0%`, `0` violações de governança e `0` citações alucinadas (resultados exclusivamente sintéticos — não substituem o PDF oficial).
- **Benchmark oficial RBAC 43 Emenda 05 (`--corpus official-approved`):** estrutura pronta em `config/evaluations/rbac43_emd05_eval_dataset.json` e `corpus_evaluator.py`, **PENDENTE DE EXECUÇÃO LOCAL NO WINDOWS** após ingestão real do `RBAC43EMD05.pdf`, revisão humana e indexação com `gemini-embedding-2` (768d Free Tier).
- **Limiar `max_vector_distance`:** mantido em **`0.85` (PROVISÓRIO)** em `config/settings.yaml`.

---

## 4. Matriz de Pendências Locais no Windows 11 do Proprietário

| Verificação | Estado no CI / VM Linux | Estado no Windows 11 Local |
| --- | :---: | :---: |
| Liberação específica de `F:\DockerRecovery\docker_data_working.vhdx` no WSL (se anexado) | Não aplicável | **PENDENTE DE AÇÃO LOCAL** |
| Docker Desktop Windows (`rbac-chromadb` + `rbac-bot` saudáveis, non-root, `10m` x `5`) | **PASS** (CI Linux) | **PENDENTE DE VALIDAÇÃO LOCAL** |
| Persistência real pós-restart (ChromaDB, SQLite, Auditoria JSONL `/app/logs`) | **PASS** (CI Linux) | **PENDENTE DE VALIDAÇÃO LOCAL** |
| Suíte `ruff` + `mypy` + `pytest` no clone `C:\Projetos\projeto-fundamentos-de-IA` | **PASS** (VM + CI) | **PENDENTE DE REVALIDAÇÃO LOCAL** |
| Presença de `.env` íntegro e `RBAC43EMD05.pdf` (`349196` bytes, SHA-256 exato) | Não acessível remotamente | **PENDENTE DE CONFIRMAÇÃO LOCAL** |
| Ingestão, revisão humana explícita, indexação Free Tier e avaliação `--corpus official-approved` | Fluxo validado em testes | **PENDENTE DE AÇÃO HUMANA LOCAL** |
| Configuração `config/access_control.yaml` + teste real no Telegram com usuário autorizado/não autorizado | Validado 100% offline com mocks | **PENDENTE DE VALIDAÇÃO LOCAL** |
| Execução real de `scripts/backup_docker_volumes.ps1` e drill de restauração | Motor Python + contratos `.ps1` testados | **PENDENTE DE VALIDAÇÃO LOCAL** |
