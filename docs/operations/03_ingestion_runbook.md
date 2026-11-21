# Runbook Operacional — Ingestão, Validação, Indexação e Consulta (Etapas 2 e 3)

Este runbook descreve o procedimento real e verificado para homologar, ingerir, validar páginas, indexar (SQLite FTS5 + ChromaDB) e executar consultas sobre regulamentos oficiais da ANAC — sem versionar PDFs nem chaves no Git.

---

## 1. Protocolo Mandatório Pré-Ingestão (Governança ANAC)

> **Regra Inviolável:** Nunca presumir que uma emenda histórica permanece vigente. Todo regulamento deve ser conferido diretamente no portal oficial da ANAC antes da ingestão. Consultas públicas **não** são normas vigentes.

1. Consulte o portal oficial da ANAC e confirme a **Emenda vigente**, a **Resolução aprovadora**, as datas (`publication_date`, `effective_date`) e a URL oficial (`official_url` / `source_url`).
2. Atualize (se necessário) o manifesto declarativo em `config/regulations_catalog.yaml`.
3. Baixe o PDF oficial localmente (fora do Git) — por exemplo, para `data/raw_pdfs/RBAC43EMD05.pdf` — ou copie-o para o volume persistente `regulatory_documents` (`/app/data/documents/raw_pdfs/`).
4. Configure o arquivo `.env` local (nunca versionado no Git) com:
   - `GEMINI_API_KEY=<sua-chave-real>`
   - `GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite` (ou outro modelo ativo configurado)
   - `GEMINI_EMBEDDING_MODEL=gemini-embedding-001`
   - `GEMINI_EMBEDDING_DIMENSIONS=768`

---

## 2. Passo a Passo Real (Docker Compose no PC Central Windows ou Linux)

### Passo 1 — Subir/reconstruir os contêineres

```bash
docker compose up -d --build
```

Verifique saúde de ambos os serviços:

```bash
docker compose ps
docker compose exec rbac-bot python -m rbac_assistant.healthcheck
```

### Passo 2 — Ingerir o PDF oficial no SQLite (SSOT)

Se o PDF foi copiado para o contêiner (por exemplo `/app/data/documents/raw_pdfs/RBAC43EMD05.pdf`):

```bash
docker compose exec rbac-bot python scripts/ingest_regulation.py \
  --document-id anac_rbac_43 \
  --pdf-path /app/data/documents/raw_pdfs/RBAC43EMD05.pdf
```

*(Alternativamente, se desejar baixar diretamente pela URL oficial HTTPS homologada em `config/regulations_catalog.yaml`, omita `--pdf-path`.)*

### Passo 3 — Inspecionar o documento ingerido e estados por página

```bash
docker compose exec rbac-bot python scripts/inspect_document.py --document-id anac_rbac_43
docker compose exec rbac-bot python scripts/inspect_document.py --document-id anac_rbac_43 --section 43.3
```

### Passo 4 — Revisar e aprovar páginas sob demanda (`review_on_demand`)

Antes de uma seção poder ser entregue como **transcrição literal validada**, **todas** as páginas que a compõem precisam de aprovação humana explícita após conferência com o PDF oficial:

```bash
# 1. Inspecionar as páginas da seção (modo somente leitura):
docker compose exec rbac-bot python scripts/validate_extraction.py \
  --document-id anac_rbac_43 \
  --section 43.3

# 2. Aprovar explicitamente após conferência com o PDF oficial:
docker compose exec rbac-bot python scripts/validate_extraction.py \
  --document-id anac_rbac_43 \
  --section 43.3 \
  --decision approve \
  --reviewer "Seu Nome" \
  --comments "Conferido integralmente com o PDF oficial da ANAC" \
  --confirm
```

> Páginas extraídas via OCR permanecem bloqueadas (`pending_manual_review`) e páginas reprovadas (`rejected`) nunca são utilizadas para fundamentar respostas.

### Passo 5 — Indexar o documento no SQLite FTS5 e no ChromaDB

```bash
docker compose exec rbac-bot python scripts/index_document.py --document-id anac_rbac_43
```

Verifique o status consolidado dos índices (chunks no SQLite, vetores no ChromaDB, versões pendentes ou defasadas):

```bash
docker compose exec rbac-bot python scripts/index_document.py --status
```

Para forçar reindexação controlada do mesmo documento:

```bash
docker compose exec rbac-bot python scripts/index_document.py --document-id anac_rbac_43 --reindex
```

Se você alterar `GEMINI_EMBEDDING_MODEL` ou `GEMINI_EMBEDDING_DIMENSIONS` no `.env`, o sistema bloqueará automaticamente a mistura de vetores incompatíveis. Reconstrua a coleção vetorial a partir do SQLite (preservando PDFs e aprovações humanas):

```bash
docker compose exec rbac-bot python scripts/index_document.py --rebuild-collection --confirm
```

### Passo 6 — Executar a primeira consulta real (sem Telegram)

```bash
# Consulta completa (busca híbrida + explicação fundamentada via Gemini):
docker compose exec rbac-bot python scripts/query_assistant.py \
  --question "Quem pode executar manutenção segundo o RBAC 43.3?"

# Consulta somente recuperação (sem chamar o modelo generativo Gemini):
docker compose exec rbac-bot python scripts/query_assistant.py \
  --question "RBAC 43.3" \
  --no-llm

# Consulta estritamente limitada a dispositivos com transcrição literal aprovada:
docker compose exec rbac-bot python scripts/query_assistant.py \
  --question "Quais os registros obrigatórios de manutenção no RBAC 43.9?" \
  --only-approved
```

---

## 3. Verificação de Qualidade e Testes Automatizados

No ambiente de desenvolvimento local (usa exclusivamente PDFs sintéticos em `tmp_path` e mocks determinísticos, com zero custo de API):

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest --cov=rbac_assistant --cov-report=term-missing
```
