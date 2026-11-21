# Runbook Operacional — Ingestão, Validação, Indexação e Consulta

Este runbook detalha a sequência operacional executável no repositório para homologar, ingerir, validar páginas, indexar (SQLite FTS5 + ChromaDB) e consultar regulamentos oficiais da ANAC. Para a hierarquia completa de fontes de dados (PDF original, SQLite, FTS5, ChromaDB e Gemini), consulte a **Seção 2 do [`README.md`](../../README.md#2-arquitetura-de-dados-sqlite-e-fontes-de-verdade)**.

> **Governança ANAC:** nunca presuma que uma emenda histórica permanece vigente sem conferência prévia no portal oficial da ANAC. Consultas públicas **não** são normas vigentes. PDFs oficiais e arquivos `.env` **nunca** são versionados no Git.

---

## Sequência Operacional Completa (10 Passos)

### Passo 1 — Atualizar a `main` após o merge do PR

```bash
git checkout main
git pull origin main
```

### Passo 2 — Configurar o `.env` local (Gemini API Free Tier Standard)

Copie `.env.example` para `.env` na raiz do repositório e preencha sua chave da Gemini Developer API gerada no Google AI Studio **sem Google Cloud Billing vinculado** (nunca versione o `.env`):

```env
GEMINI_API_KEY=<sua-chave-gemini-api-free-tier>
GEMINI_GENERATION_MODEL=gemini-4-hydrogen
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMENSIONS=768
GEMINI_ALLOW_PAID_MODELS=false
CHROMA_HOST=chromadb
CHROMA_PORT=8000
ENVIRONMENT=local
LOG_LEVEL=INFO
```

> **Proteção de custos e privacidade (Free Tier):**
> - `GEMINI_ALLOW_PAID_MODELS=false` impede o uso acidental de modelos exclusivos do tier pago (ex.: `gemini-4-neon`, `gemini-4-carbon`, `gemini-4-argon`). Alternativa gratuita configurável: `gemini-3.8-flash`.
> - A validação no código não substitui a configuração de conta no Google Cloud; mantenha o projeto da API Key sem faturamento vinculado para garantia de custo zero.
> - No Free Tier, processe **apenas documentos regulatórios públicos oficiais da ANAC** (nunca dados pessoais ou confidenciais).

### Passo 3 — Inicializar a stack Docker

```bash
docker compose up -d --build
docker compose ps
docker compose exec rbac-bot python -m rbac_assistant.healthcheck
```

Verifique que ambos os contêineres (`rbac-chromadb` e `rbac-bot`) reportam status `healthy` operando como non-root (UID/GID `10001`).

### Passo 4 — Adicionar o PDF original no caminho persistente correto

Copie o PDF oficial conferido na ANAC (ex.: `RBAC43EMD05.pdf`) para dentro do volume persistente `regulatory_documents` montado em `/app/data/documents`:

```bash
docker compose cp ./RBAC43EMD05.pdf rbac-bot:/app/data/documents/raw_pdfs/RBAC43EMD05.pdf
```

*(O banco operacional SQLite ficará no mesmo volume persistente em `/app/data/documents/catalog.sqlite3`.)*

### Passo 5 — Executar a ingestão no SQLite

```bash
docker compose exec rbac-bot python scripts/ingest_regulation.py \
  --document-id anac_rbac_43 \
  --pdf-path /app/data/documents/raw_pdfs/RBAC43EMD05.pdf
```

### Passo 6 — Validar e inspecionar a extração

Inspecione o documento ingerido, a contagem de páginas/unidades e o estado inicial de validação:

```bash
docker compose exec rbac-bot python scripts/inspect_document.py --document-id anac_rbac_43
docker compose exec rbac-bot python scripts/inspect_document.py --document-id anac_rbac_43 --section 43.3
```

### Passo 7 — Revisar e aprovar as páginas necessárias (`review_on_demand`)

Antes que uma seção seja apresentada como **transcrição literal validada**, todas as páginas que a compõem exigem revisão humana explícita após conferência com o PDF original:

```bash
# Modo somente inspeção (exibe texto original extraído página a página):
docker compose exec rbac-bot python scripts/validate_extraction.py \
  --document-id anac_rbac_43 \
  --section 43.3

# Registro explícito de aprovação após conferência com o PDF oficial:
docker compose exec rbac-bot python scripts/validate_extraction.py \
  --document-id anac_rbac_43 \
  --section 43.3 \
  --decision approve \
  --reviewer "Seu Nome" \
  --comments "Conferido integralmente com o PDF oficial da ANAC" \
  --confirm
```

> Páginas extraídas via OCR permanecem bloqueadas (`pending_manual_review`) e páginas reprovadas (`rejected`) nunca são utilizadas para fundamentar respostas.

### Passo 8 — Indexar os chunks no SQLite FTS5 e no ChromaDB

Gera chunks estruturais determinísticos no SQLite (`normative_chunks`), atualiza o índice lexical FTS5 (`normative_chunks_fts`) e envia vetores `gemini-embedding-2` (768d) para a coleção versionada `regulatory_chunks_ge2_768_v1` no ChromaDB:

```bash
docker compose exec rbac-bot python scripts/index_document.py --document-id anac_rbac_43
```

### Passo 9 — Inspecionar o status da indexação

Verifique chunks no SQLite, vetores no ChromaDB, versões não indexadas, versões defasadas e compatibilidade de modelo/dimensionalidade:

```bash
docker compose exec rbac-bot python scripts/index_document.py --status
```

### Passo 10 — Executar consulta de teste (sem Telegram)

```bash
# Consulta completa (busca híbrida + explicação fundamentada via Gemini):
docker compose exec rbac-bot python scripts/query_assistant.py \
  --question "Quem pode executar manutenção segundo o RBAC 43.3?"

# Consulta somente recuperação (SSOT SQLite + FTS5, sem chamada generativa ao Gemini):
docker compose exec rbac-bot python scripts/query_assistant.py \
  --question "RBAC 43.3" \
  --no-llm

# Consulta com diagnóstico completo do pipeline RAG (FTS5, ChromaDB, RRF, revisão e citações):
docker compose exec rbac-bot python scripts/query_assistant.py \
  --question "Quais os registros obrigatórios de manutenção no RBAC 43.9?" \
  --only-approved \
  --debug
```

---

## Operações de Manutenção Controlada

- **Reindexar um documento específico:**
  ```bash
  docker compose exec rbac-bot python scripts/index_document.py --document-id anac_rbac_43 --reindex
  ```
- **Reconstruir a coleção vetorial a partir do SQLite** (após alterar modelo de embeddings, dimensionalidade ou estratégia de instrução — preserva PDFs, SQLite e aprovações humanas):
  ```bash
  docker compose exec rbac-bot python scripts/index_document.py --rebuild-collection --confirm
  ```
- **Executar benchmark quantitativo sintético controlado (36 consultas, 0 chamadas externas):**
  ```bash
  uv run python scripts/evaluate_rag.py
  ```
- **Executar suíte automatizada de qualidade (usa apenas fixtures sintéticas e mocks, sem custo de API):**
  ```bash
  uv run ruff check src tests scripts
  uv run mypy src
  uv run pytest --cov=rbac_assistant --cov-report=term-missing
  ```
