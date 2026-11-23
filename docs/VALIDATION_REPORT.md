# Relatório de Validação Técnica — Consolidação Atualizada (23/11/2026)

> **Princípio de veracidade:** todos os números abaixo foram efetivamente executados e observados na VM Linux do Antigravity, no GitHub Actions CI ou reportados pelo proprietário no Windows 11. Nenhuma validação Docker Desktop Windows, ingestão oficial do PDF ou teste real de Telegram/Gemini foi simulada ou declarada concluída sem execução real no host local.
>
> **Isolamento:** todas as execuções de testes operam com `RBAC_DISABLE_DOTENV=1`, `RBAC_BLOCK_REAL_GEMINI_API=1`, `RBAC_BLOCK_REAL_TELEGRAM_API=1` e `PTB_TIMEDELTA=1` (zero chamadas reais às APIs externas e zero avisos de depreciação suprimidos).

---

## 1. Validações Confirmadas no GitHub Actions (CI Linux + Docker Real)

| PR / Ref | Merge Commit / HEAD | CI Run ID | `Lint, Typecheck & Pytest (3.12)` | `Docker Non-Root, Heartbeat & Persistence Smoke Test` |
| --- | --- | :---: | :---: | :---: |
| PR #10 (`chore/docker-log-rotation`) | `292c11667f7fef145add130a7ff07eec9747a2bf` | `45270448546` (`main`) | **SUCCESS** (`53s`) | **SUCCESS** (`1m29s`) |
| PR #11 (`chore/docker-runtime-recovery`) | `5112bd5048175a932e746b21225424eeed39568f` | `45271482437` (`main`) | **SUCCESS** (`1m10s`) | **SUCCESS** (`1m33s`) |
| PR #12 (`feat/persistent-audit-adapter`) | `571b0b67c3ef737972ed4008bedf66a0aa6c8f2b` | `45277532713` (`main`) | **SUCCESS** (`1m13s`) | **SUCCESS** (`1m33s`, incl. persistência JSONL `/app/logs`) |
| PR #13 (`feat/etapa-4-telegram-bot`) | `65dc60f9e46e29ca83e8fa632d5f691b2fb66014` | `45279737245` (`main`) | **SUCCESS** (`1m12s`) | **SUCCESS** (`1m25s`, incl. mount `./config:/app/config:ro` + standby) |
| PR #14 (`feat/etapa-5-rag-evaluation-quality`) | `ff74031ee3eabc6eaa37175308aec2b5a461b4ca` | `45281271249` (`main`) | **SUCCESS** (`1m21s`) | **SUCCESS** (`1m34s`) |
| PR #15 (`feat/backup-restore-and-final-docs`) | `c607a7f0e2972374bec45846dfecd82fbc63261a` | `45282717315` (`main`) | **SUCCESS** (`1m11s`) | **SUCCESS** (`1m34s`) |
| PR #16 (`fix/telegram-timedelta-compat`) | `3558228b659908c8cb28c7d4b3ea80dd8f91f803` | `45288149183` (`main`) | **SUCCESS** (`1m12s`, `0 warnings`) | **SUCCESS** (`1m36s`, incl. `PTB_TIMEDELTA=1` no contêiner `rbac-bot`) |

---

## 2. Cobertura de Qualidade e Suítes Automatizadas (VM Linux Debian 12, Python 3.12.16)

- **`uv sync --frozen --check`:** íntegro (`116` pacotes consistentes com `uv.lock`, incluindo `python-telegram-bot==22.8`).
- **`uv run ruff check src tests scripts`:** **`All checks passed!`** (0 erros).
- **`uv run mypy src` (strict):** **`Success: no issues found`** (0 erros em todos os `91` módulos de `src/rbac_assistant/`).
- **Warnings (`PTBDeprecationWarning`):** **`0 warnings`** após ativação nativa de `PTB_TIMEDELTA=1` (sem filtros ou supressão global).
- **Cobertura global (`pytest --cov=rbac_assistant`):** mantida **`>= 91%`** ao longo dos PRs #12 a #16 e `feat/telegram-public-access-mode`.
- **Suítes críticas cobertas 100% offline:**
  - `tests/unit/test_access_control.py`: modo público (`public_access: true`, `default_role: USER`), 1º e 2º administradores (`ADMIN`), não bloqueio de usuário desconhecido no modo público, bloqueio de escalada para `ADMIN`, desabilitação explícita via `public_access: false` e legado `deny_by_default: true`, fail-safe em arquivo ausente/malformado/incompatível (`default_role: ADMIN` rejeitado), IDs inválidos e `UserRateLimiter` (por usuário + rajada + limite global).
  - `tests/unit/test_telegram_bot.py`: todos os 9 comandos (incluindo `/id`) + texto livre, `/status` público sanitizado vs. diagnóstico `ADMIN` para os 2 administradores, `/id` retornando somente o ID do solicitante sem vazamento em logs públicos, rejeição de grupos/supergrupos/canais (`allow_group_chats=False`), isolamento de contexto entre dois usuários, rate limit global e por usuário, timeout de comandos, normalização de `RetryAfter` (`int`, `float`, `timedelta`), backoff/retry na inicialização, bloqueio real da API Telegram (`RBAC_BLOCK_REAL_TELEGRAM_API=1`) e `0` `PTBDeprecationWarning`.
  - `tests/evals/test_regulatory_fidelity_regression.py`: exclusão determinística de páginas pendentes/preliminares/reprovadas no escopo estrito (`only_approved_literal=True`), recusa fundamentada fora do escopo ou sem evidência (`llm.calls == 0`), detecção de *prompt injection*/exfiltração/escalada de privilégio, redação de segredos na saída, descarte de citações fabricadas pelo `CitationValidator` e preservação de `max_vector_distance=0.85`.
  - `tests/unit/test_jsonl_audit_adapter.py`, `tests/unit/test_message_splitter.py`, `tests/unit/test_backup_restore.py` e `tests/unit/test_chroma_infrastructure.py`.

---

## 3. Avaliação Quantitativa Sintética vs. Avaliação Oficial Real

- **Benchmark sintético (`scripts/evaluate_rag.py`, 40 consultas fictícias, embedder determinístico `768d`):** `hybrid_rrf` mantém `Rec@5 = 100.0%`, `Prec@5 = 85.7%`, `Prec@1 = 100.0%`, `MRR@10 = 1.000`, `nDCG@10 = 1.000`, `Abst = 100.0%`, `0` violações de governança e `0` citações alucinadas (resultados exclusivamente sintéticos — não substituem o PDF oficial).
- **Benchmark oficial RBAC 43 Emenda 05 (`--corpus official-approved`):** estrutura pronta em `config/evaluations/rbac43_emd05_eval_dataset.json` e `corpus_evaluator.py`, **PENDENTE DE EXECUÇÃO LOCAL NO WINDOWS** após ingestão real do `RBAC43EMD05.pdf`, revisão humana e indexação com `gemini-embedding-2` (768d Free Tier).
- **Limiar `max_vector_distance`:** mantido em **`0.85` (PROVISÓRIO)** em `config/settings.yaml`.

---

## 4. Matriz de Pendências Locais no Windows 11 do Proprietário

| Verificação | Estado no CI / VM Linux | Estado no Windows 11 Local |
| --- | :---: | :---: |
| Suíte Python (`ruff`, `mypy`, `pytest` seq/par) | **PASS** (VM + CI, `0 warnings`) | Validado em `c607a7f` (`138 passed`, `91%`); **PENDENTE** revalidar pós-pull (`0 warnings` + modo público) |
| Liberação específica de `F:\DockerRecovery\docker_data_working.vhdx` no WSL (se anexado) | Não aplicável | **PENDENTE DE AÇÃO LOCAL** |
| Docker Desktop Windows (`rbac-chromadb` + `rbac-bot` saudáveis, non-root, `10m` x `5`, `PTB_TIMEDELTA=1`) | **PASS** (CI Linux) | **PENDENTE DE VALIDAÇÃO LOCAL** |
| Persistência real pós-restart (ChromaDB, SQLite, Auditoria JSONL `/app/logs`) | **PASS** (CI Linux) | **PENDENTE DE VALIDAÇÃO LOCAL** |
| Presença de `.env` íntegro e `RBAC43EMD05.pdf` (`349196` bytes, SHA-256 exato) | Não acessível remotamente | **PENDENTE DE CONFIRMAÇÃO LOCAL** |
| Ingestão, revisão humana explícita, indexação Free Tier e avaliação `--corpus official-approved` | Fluxo validado em testes | **PENDENTE DE AÇÃO HUMANA LOCAL** |
| Descoberta dos 2 IDs via `/id`, criação de `config/access_control.yaml` (modo público + 2 admins) e teste real no Telegram | Validado 100% offline com mocks | **PENDENTE DE VALIDAÇÃO LOCAL** |
| Execução real de `scripts/backup_docker_volumes.ps1` e drill de restauração | Motor Python + contratos `.ps1` testados | **PENDENTE DE VALIDAÇÃO LOCAL** |
