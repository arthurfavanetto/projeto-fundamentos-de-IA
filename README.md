# RBAC AI Assistant (ANAC)

Assistente normativo de Inteligência Artificial especializado em **Regulamentos Brasileiros da Aviação Civil (RBAC — ANAC)**, inicialmente focado no **RBAC 43** (*Manutenção, Manutenção Preventiva, Reconstrução e Alteração*), operado via **Telegram (Long Polling)** e executado em contêineres **Docker**.

> **Nota terminológica:** Neste projeto, `RBAC` refere-se exclusivamente a **Regulamento Brasileiro da Aviação Civil (ANAC)**. O controle de acesso operacional do bot utiliza `access_control` e `operator_roles` (`ADMIN` / `USER`).

---

## Arquitetura e Princípios

O sistema adota **Arquitetura Hexagonal (*Ports and Adapters*)** com `src-layout` em Python 3.12+ gerenciado por `uv`:

- **Fidelidade Literal Garantida (SSOT SQLite):** O texto oficial das normas é armazenado em SQLite (WAL/ACID) no volume `regulatory_documents` com validação por página (`review_on_demand`) e entregue literalmente em Python apenas quando todas as páginas da seção estão aprovadas — **o Google Gemini nunca reescreve nem gera o texto normativo**.
- **RAG Jurídico-Normativo Híbrido (Estrutural + FTS5 + ChromaDB + RRF):** Combina busca estrutural determinística (ex.: `RBAC 43.3`, `43.9(a)`, `Art. 1º`), busca lexical SQLite **FTS5** (BM25, tolerante a acentos) e busca semântica vetorial no **ChromaDB 1.5.9**, fundidas via **Reciprocal Rank Fusion (RRF)** e hidratadas a partir do SSOT SQLite.
- **Saída Estruturada em 4 Blocos:** Separa visualmente `TEXTO NORMATIVO` (literal validado do SQLite), `EXPLICAÇÃO` (gerada por IA via SDK oficial `google-genai`), `REFERÊNCIA` (regulamento, emenda, vigência, seção, página impressa ANAC e página física PDF) e `OBSERVAÇÃO` (avisos de revisão, bloqueio OCR/reprovação, vigência ou degradação).
- **Segurança Python-First:** Defesa contra *Prompt Injection* direta/indireta (`PromptGuard`), redação de segredos em logs e erros (`secret_redactor`), controle de compatibilidade de modelos/dimensões de embeddings e allowlist Telegram (Etapa 4).
- **Operação Portátil:** Dois contêineres Docker non-root (UID/GID `10001`: `rbac-bot` + `chromadb`) em rede interna isolada, operando tanto no **PC Central Windows (Docker Desktop)** quanto em **VPS Linux** sem alteração de código.

---

## Estrutura do Repositório

```text
.
├── config/                    # Configurações não-sensíveis, catálogo ANAC e prompts versionados
│   └── prompts/               # Templates versionados do motor RAG (Markdown)
├── docs/
│   ├── architecture/          # Especificação arquitetural, ADRs e gramática ANAC
│   ├── operations/            # Guias Windows, migração VPS e runbook de ingestão/indexação
│   ├── evaluations/           # Protocolo de fidelidade normativa e grounding IA
│   └── legacy/                # Rascunhos históricos arquivados
├── scripts/                   # CLIs: ingestão, inspeção, validação, indexação e consulta RAG
├── src/rbac_assistant/        # Código-fonte (Arquitetura Hexagonal, src-layout)
│   ├── config/                # Pydantic Settings e logging estruturado
│   ├── domain/                # Entidades normativas, RAG e Ports (zero deps externas)
│   ├── application/           # DTOs, Serviços (Catalog, Ingestion, Indexing, Chat) e Use Cases
│   ├── security/              # PromptGuard, secret_redactor e (Etapa 4) access control/audit
│   ├── rag/                   # Chunking estrutural, SectionResolver, HybridRetriever (RRF) e Formatter
│   ├── infrastructure/        # Adaptadores SQLite+FTS5, ChromaDB 1.5.9, Google GenAI e pdfplumber
│   └── interfaces/telegram/   # (Etapa 4) Bot Long Polling, middleware, handlers, splitter
└── tests/                     # Testes unitários, integração e evals de fidelidade (fixtures sintéticas)
```

---

## Documentação

- [Arquitetura do Sistema](docs/architecture/01_system_architecture.md)
- [Decisões Arquiteturais (ADRs)](docs/architecture/02_adr_decisions.md)
- [Estrutura Normativa ANAC e Metadados](docs/architecture/03_anac_normative_structure.md)
- [Setup Local — PC Central Windows](docs/operations/01_local_windows_setup.md)
- [Migração para VPS Linux](docs/operations/02_vps_linux_migration.md)
- [Runbook de Ingestão, Validação, Indexação e Consulta](docs/operations/03_ingestion_runbook.md)
- [Protocolo de Fidelidade Normativa](docs/evaluations/normative_fidelity_protocol.md)

---

## Configuração Inicial (Segurança)

Arquivos com segredos, Telegram User IDs reais ou PDFs oficiais **nunca são versionados no Git** (protegidos pelo `.gitignore`):

1. Copie `.env.example` para `.env` e preencha `GEMINI_API_KEY`, `GEMINI_GENERATION_MODEL` (padrão atual: `gemini-3.5-flash-lite`), `GEMINI_EMBEDDING_MODEL` (padrão atual: `gemini-embedding-001`) e, na Etapa 4, `TELEGRAM_BOT_TOKEN`.
2. Copie `config/access_control.example.yaml` para `config/access_control.yaml` quando configurar o bot Telegram.
3. Siga [`docs/operations/03_ingestion_runbook.md`](docs/operations/03_ingestion_runbook.md) para adicionar o PDF oficial do RBAC 43, ingerir, validar páginas, indexar (FTS5 + ChromaDB) e executar sua primeira consulta real.

---

## Roadmap de Implementação

- [x] **Etapa 0 — Arquitetura e Documentação:** Estrutura `src-layout`, ADRs, guias operacionais, `.gitignore`, `.dockerignore` e templates de configuração.
- [x] **Etapa 1 — Fundação Python e Docker:** `pyproject.toml`, `settings.py`, fundação hexagonal, `Dockerfile` (multi-stage non-root), `docker/chromadb.Dockerfile` (`1.5.9` non-root) e `docker-compose.yml` (`rbac-bot` + `chromadb`).
- [x] **Etapa 2 — Motor de Ingestão Documental Regulatória (Multidocumento + RBAC 43):** Catálogo SQLite (WAL/ACID), `pdfplumber` (MIT), OCR opcional desacoplado, parser estrutural ANAC + fallback genérico, validação híbrida (`review_on_demand`) e CLIs de ingestão/revisão.
- [x] **Etapa 3 — Gemini e Motor RAG Híbrido:** SDK oficial `google-genai`, SQLite FTS5 (`unicode61 remove_diacritics 2`), ChromaDB 1.5.9 com rastreabilidade e guarda de compatibilidade de embeddings, fusão RRF, `PromptGuard`, `NormativeResponseFormatter` e CLIs `scripts/index_document.py` e `scripts/query_assistant.py`.
- [ ] **Etapa 4 — Interface Telegram e Controle de Acesso:** Bot Long Polling, middleware `ADMIN`/`USER`, comandos (`/start`, `/help`, `/status`, `/rbac`, `/buscar`, `/explicar`, `/documentos`, `/versao`) e `MessageSplitter`.
- [ ] **Etapa 5 — Qualidade, Evals e Operação Contínua:** Suíte completa `pytest`, dataset áureo de fidelidade normativa e backup de volumes.