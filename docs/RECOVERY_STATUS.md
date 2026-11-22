# Estado de Recuperação — Incidente Docker (22/11/2026)

> **Princípio de veracidade:** este documento registra apenas fatos comprovados no repositório Git, no GitHub e na validação executada em 22/11/2026. Nenhum volume antigo foi declarado recuperado sem evidência.

---

## 1. Resumo do Incidente

Em **22/11/2026**, foi executado indevidamente o comando destrutivo `docker compose down -v` no host Windows 11 do proprietário, removendo containers, redes do projeto e os três volumes nomeados do RBAC AI Assistant:

- `projeto-fundamentos-de-ia_audit_logs`
- `projeto-fundamentos-de-ia_regulatory_documents`
- `projeto-fundamentos-de-ia_chroma_data`

### Investigação forense registrada (host Windows / WSL)
- Docker Desktop foi encerrado após o incidente.
- Uma cópia de trabalho do VHDX (`F:\DockerRecovery\docker_data_working.vhdx`, tamanho `47.422.898.176` bytes, SHA-256 `7C13330841763A774D7F2C7273877A7AC2DBAA826C9E3019E6010EA0A4F75060`) foi anexada ao `Ubuntu-24.04` WSL em modo somente leitura (`/dev/sdd`, `blockdev --getro /dev/sdd == 1`).
- O banco `metadata.db` do Docker ainda registrava referência a `regulatory_documents`, mas os diretórios dos três volumes do projeto não apareceram na listagem consultada.
- **Conclusão operacional:** os volumes antigos **não** foram recuperados. Como a auditoria pré-incidente já confirmara que ainda não havia documentos oficiais RBAC ingeridos nem aprovações humanas registradas nos volumes antigos, a reconstrução com **volumes novos e vazios** não acarreta perda de código nem de base normativa já validada.

---

## 2. Proteção de Backups VHDX, Credenciais e PDF

| Ativo local (host Windows) | Localização conhecida | Política obrigatória |
| --- | --- | --- |
| VHDX original Docker | `C:\Users\Desktop\AppData\Local\Docker\wsl\disk\docker_data.vhdx` | **NÃO tocar, excluir ou reparar** |
| Backup VHDX | `D:\DockerRecovery\docker_data_recovery_2026-11-22.vhdx` | **NÃO tocar, excluir ou renomear** |
| Cópia de trabalho VHDX | `F:\DockerRecovery\docker_data_working.vhdx` | **NÃO iniciar Docker Desktop enquanto estiver anexada como `/dev/sdd` no WSL** |
| WSL Ubuntu forense | `Ubuntu-24.04` em `F:\WSL\Ubuntu-24.04` | **NÃO executar `wsl --unregister` nem desinstalar** |
| Volumes de outros projetos | `n8n_data`, `ollama`, `open-webui` | **Preservar intactos — nunca reutilizar ou remover** |
| Arquivo `.env` local | `C:\Projetos\projeto-fundamentos-de-IA\.env` | **Preservar intacto (`GEMINI_API_KEY` + `TELEGRAM_BOT_TOKEN`), fora do Git** |
| PDF oficial RBAC 43 Emenda 05 | `C:\RBAC_AI_Assistant\source_pdfs\RBAC43EMD05.pdf` e `data/raw_pdfs/RBAC43EMD05.pdf` | **Preservar intacto, fora do Git** (SHA-256 `5026d2da87fc3f55c61a6b5a20dd1f50d52a1367a8550897f3864074e1a46472`) |

---

## 3. Estado Reconciliado do Código e Ambiente Python

- **Repositório GitHub:** `arthurfavanetto/projeto-fundamentos-de-IA`
- **Último estado integrado em `origin/main`:** `d1d6af150b0a20292ae7f91e33da0b91146573ca`
  - **PR #4** (`02d68eb`): Etapa 0 — Baseline arquitetural e ADRs
  - **PR #5** (`7e096ee`): Etapa 1 — Fundação Python hexagonal e stack Docker non-root (`chromadb/chroma:1.5.9`)
  - **PR #6** (`5ebc919`): Etapa 2 — Motor de ingestão regulatória (SQLite SSOT, `pdfplumber`, validação híbrida)
  - **PR #7** (`acb600d`): Etapa 3 — Gemini Free Tier (`gemini-4-hydrogen` + `gemini-embedding-2` 768d), SQLite FTS5, ChromaDB e RAG híbrido (RRF)
  - **PR #9** (`4d4673bc19fecf661ce1a048acbe9abf5de7cd4b`): Compatibilidade Windows, CLI `--json` puro (`stderr`), isolamento `.env`, bloqueio de chamadas Gemini reais em testes/subprocessos, SQLite `:memory:`, dedup N+1 e `pytest-xdist`
  - **PR #8** (`d1d6af150b0a20292ae7f91e33da0b91146573ca`): Cache temporário em memória de embeddings de consulta no benchmark (`1.286` → `46` chamadas HTTP simuladas)
- **Branches preservadas no remoto:** `main`, `docs/etapa-0-architecture-baseline`, `feat/etapa-1-foundation-docker`, `feat/etapa-2-rbac43-ingestion`, `feat/etapa-3-gemini-hybrid-rag`, `fix/windows-test-compatibility`, `perf/eval-query-embedding-cache`, `arthurfavanetto-patch-2`.
- **Ambiente Python / `uv`:** `Python 3.12.16`, `uv sync --frozen --check` íntegro (`115` pacotes consistentes com `uv.lock`).
- **Qualidade e testes revalidados em `d1d6af1` (ambiente Linux VM, zero chamadas Gemini reais):**
  - `ruff check src tests scripts`: aprovado.
  - `mypy src` (strict): aprovado em `82` arquivos.
  - Suíte completa: **93/93 testes aprovados**, **91% de cobertura** (`4.741` statements, `442` missed) idêntica nos modos sequencial e paralelo (`pytest-xdist`).
  - CLI offline (`scripts/inspect_document.py`, `scripts/query_assistant.py --no-llm --json/--debug`): validada em base vazia com recusa segura.
  - PDF local no workspace (`data/raw_pdfs/RBAC43EMD05.pdf`): presente e íntegro (SHA-256 exato confirmado).

---

## 4. Volumes a Recriar no Host Windows e Pendências Reais

### Volumes que serão criados **novos e vazios** no próximo `docker compose up -d --build` (no Windows)
| Volume definido em `docker-compose.yml` | Nome composto esperado | Destino no contêiner | Estado atual no Windows |
| --- | --- | --- | --- |
| `chroma_data` | `projeto-fundamentos-de-ia_chroma_data` | `/data` (`rbac-chromadb`) | **Pendente de recriação no host Windows** |
| `regulatory_documents` | `projeto-fundamentos-de-ia_regulatory_documents` | `/app/data/documents` (`rbac-bot`) | **Pendente de recriação no host Windows** |
| `audit_logs` | `projeto-fundamentos-de-ia_audit_logs` | `/app/logs` (`rbac-bot`) | **Pendente de recriação no host Windows** |

### Limitações desta execução (VM Linux remota vs. Host Windows 11)
1. **Docker Engine / Docker Desktop no Windows:** não acessível a partir da VM Linux do Antigravity (que também não possui o binário `docker` instalado). Os testes de Docker (non-root UID/GID 10001, heartbeat, isolamento de rede e persistência) estão validados no GitHub Actions CI Run `45068432151` sobre `d1d6af1`, mas a recriação local dos containers e volumes no Windows depende de liberação segura do disco `/dev/sdd` pelo proprietário.
2. **Anexação do VHDX `/dev/sdd` no WSL:** não foi desmontada automaticamente. O proprietário deve conferir se `/dev/sdd` ainda está anexado e liberá-lo explicitamente antes de abrir o Docker Desktop (ver [`docs/DOCKER_SETUP.md`](./DOCKER_SETUP.md)).
3. **Etapa 4 (Bot Telegram):** ainda não implementada no código (`src/rbac_assistant/interfaces/telegram/__init__.py` é placeholder). O container `rbac-bot` executa hoje apenas a fundação da Etapa 1 (`src/rbac_assistant/main.py`).
4. **Ingestão do RBAC 43:** pendente de execução controlada após a recriação dos volumes no Docker Windows e revisão humana explícita das páginas.
