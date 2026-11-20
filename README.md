# RBAC AI Assistant (ANAC)

Assistente normativo de Inteligência Artificial especializado em **Regulamentos Brasileiros da Aviação Civil (RBAC — ANAC)**, inicialmente focado no **RBAC 43** (*Manutenção, Manutenção Preventiva, Reconstrução e Alteração*), operado via **Telegram (Long Polling)** e executado em contêineres **Docker**.

> **Nota terminológica:** Neste projeto, `RBAC` refere-se exclusivamente a **Regulamento Brasileiro da Aviação Civil (ANAC)**. O controle de acesso operacional do bot utiliza `access_control` e `operator_roles` (`ADMIN` / `USER`).

---

## Arquitetura e Princípios

O sistema adota **Arquitetura Hexagonal (*Ports and Adapters*)** com `src-layout` em Python 3.12+ gerenciado por `uv`:

- **Fidelidade Literal Garantida (SSOT):** O texto oficial das normas é armazenado em JSON canônico validado (`regulatory_documents`) e entregue literalmente em Python — **o Google Gemini nunca reescreve nem gera o texto normativo**.
- **RAG Jurídico-Normativo Híbrido:** Prioriza busca determinística por identificador de seção (ex.: `43.3`, `43.9(a)`) combinada com busca semântica no **ChromaDB** filtrada por emenda vigente.
- **Saída Dual Obrigatória:** Toda resposta separa visualmente `TEXTO OFICIAL`, `FUNDAMENTAÇÃO` (regulamento, emenda, seção, página impressa ANAC e página física PDF) e `EXPLICAÇÃO` (Google Gemini via SDK oficial `google-genai`).
- **Segurança Python-First:** Allowlist determinística por `Telegram User ID` (*deny-by-default*), *rate limiting*, redação de segredos em logs JSONL e defesa contra *Prompt Injection* indireta.
- **Operação Portátil:** Dois contêineres Docker (`rbac-bot` + `chromadb`) em rede interna isolada, operando tanto no **PC Central Windows (Docker Desktop)** quanto em **VPS Linux** sem alteração de código.

---

## Estrutura do Repositório

```text
.
├── config/                    # Configurações não-sensíveis, catálogo ANAC e templates
├── docs/
│   ├── architecture/          # Especificação arquitetural, ADRs e gramática ANAC
│   ├── operations/            # Guias Windows, migração VPS e runbook de ingestão
│   ├── evaluations/           # Protocolo de fidelidade normativa e grounding IA
│   └── legacy/                # Rascunhos históricos arquivados
├── scripts/                   # Utilitários CLI (verificação ANAC, ingestão, backup)
├── src/rbac_assistant/        # Código-fonte (Arquitetura Hexagonal, src-layout)
│   ├── config/                # Pydantic Settings e logging estruturado
│   ├── domain/                # Entidades normativas e Ports (zero deps externas)
│   ├── application/           # DTOs e Use Cases
│   ├── security/              # Access control, rate limit, prompt guard, audit log
│   ├── rag/                   # Ingestão PDF, parser ANAC, chunking, retrieval, prompts
│   ├── infrastructure/        # Adaptadores ChromaDB, Google GenAI, SSOT, scheduler
│   └── interfaces/telegram/   # Bot Long Polling, middleware, handlers, splitter
└── tests/                     # Testes unitários, integração e evals de fidelidade
```

---

## Documentação

- [Arquitetura do Sistema](docs/architecture/01_system_architecture.md)
- [Decisões Arquiteturais (ADRs)](docs/architecture/02_adr_decisions.md)
- [Estrutura Normativa ANAC e Metadados](docs/architecture/03_anac_normative_structure.md)
- [Setup Local — PC Central Windows](docs/operations/01_local_windows_setup.md)
- [Migração para VPS Linux](docs/operations/02_vps_linux_migration.md)
- [Runbook de Verificação ANAC e Ingestão](docs/operations/03_ingestion_runbook.md)
- [Protocolo de Fidelidade Normativa](docs/evaluations/normative_fidelity_protocol.md)

---

## Configuração Inicial (Segurança)

Arquivos com segredos, Telegram User IDs reais ou PDFs oficiais **nunca são versionados no Git** (protegidos pelo `.gitignore`):

1. Copie `.env.example` para `.env` e preencha `TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY` e os modelos Gemini validados na conta.
2. Copie `config/access_control.example.yaml` para `config/access_control.yaml` e adicione os Telegram User IDs autorizados (`admins` e `users`).
3. Siga [`docs/operations/03_ingestion_runbook.md`](docs/operations/03_ingestion_runbook.md) para homologar o PDF oficial do RBAC 43 no `config/regulations_catalog.yaml`.

---

## Roadmap de Implementação

- [x] **Etapa 0 — Arquitetura e Documentação:** Estrutura `src-layout`, ADRs, guias operacionais, `.gitignore`, `.dockerignore` e templates de configuração.
- [x] **Etapa 1 — Fundação Python e Docker:** `pyproject.toml`, `settings.py`, fundação hexagonal, `Dockerfile` (multi-stage non-root), `docker/chromadb.Dockerfile` (`1.5.9` non-root) e `docker-compose.yml` (`rbac-bot` + `chromadb`).
- [ ] **Etapa 2 — Ingestão e Estruturação do RBAC 43:** Protocolo ANAC, extração nativa + OCR condicional, parser hierárquico ANAC e `Canonical Store` (SSOT).
- [ ] **Etapa 3 — Gemini e Motor RAG Híbrido:** Adaptador `google-genai`, ChromaDB com 18 metadados, busca determinística + semântica e `ResponseFormatter` dual.
- [ ] **Etapa 4 — Interface Telegram e Controle de Acesso:** Bot Long Polling, middleware `ADMIN`/`USER`, comandos (`/start`, `/help`, `/status`, `/rbac`, `/buscar`, `/explicar`, `/documentos`, `/versao`) e `MessageSplitter`.
- [ ] **Etapa 5 — Qualidade, Evals e Operação Contínua:** Suíte completa `pytest`, dataset áureo de fidelidade normativa e backup de volumes.