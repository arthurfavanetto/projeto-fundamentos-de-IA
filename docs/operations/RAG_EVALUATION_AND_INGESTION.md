# Guia Operacional — Ingestão do RBAC 43, Revisão Humana, Indexação e Avaliação Real do RAG

Este documento consolida o fluxo oficial de ingestão do **`RBAC43EMD05.pdf`** (fora do Git), revisão humana página a página, indexação Free Tier (`gemini-embedding-2` 768d) e avaliação/calibração reproduzível do motor RAG.

---

## 1. Separação Metodológica (Sintético vs. Corpus Oficial Real)

| Nível | Comando | Corpus | Embeddings | Finalidade |
| --- | --- | --- | --- | --- |
| **1. Sintético CI / Offline** | `uv run python scripts/evaluate_rag.py` | 40 consultas sintéticas (`tests/evals/rag_eval_dataset.json`) | Determinístico local (`768d`) | Regressão mecânica de recuperação, filtros, RRF, reranker, governança e `CitationValidator` sem custo de API. |
| **2. Oficial — Somente Lexical/Estrutural** | `docker compose exec -T rbac-bot python scripts/evaluate_rag.py --corpus official-approved --embedding-backend deterministic` | Dataset oficial RBAC 43 Emenda 05 (`config/evaluations/rbac43_emd05_eval_dataset.json`) | Sem API Gemini | Valida recall estrutural/FTS5 e recusas seguras exclusivamente sobre páginas já aprovadas manualmente no SQLite real. |
| **3. Oficial — Completo + Calibração de Limiar** | `docker compose exec -T rbac-bot python scripts/evaluate_rag.py --corpus official-approved --embedding-backend gemini-embedding-2 --json` | Dataset oficial RBAC 43 Emenda 05 sobre SQLite + ChromaDB reais | **`gemini-embedding-2` (768d, Free Tier)** | Mede Recall@5, Precision@5, Precision@1, MRR@10, nDCG@10, fidelidade de citações, recusa segura e gera recomendação reproduzível para `max_vector_distance` (atualmente **`0.85` provisório**). |

> **Regra de veracidade:** métricas do nível 1 (sintético) **nunca** substituem nem devem ser apresentadas como métricas do documento oficial da ANAC. O comando `--corpus official-approved` recusa executar se `anac_rbac_43` não estiver ingerido ou se possuir `0` páginas aprovadas manualmente.

---

## 2. Ingestão, Inspeção e Revisão Humana do `RBAC43EMD05.pdf` (Windows 11 PowerShell)

### 2.1. Validar integridade do PDF local (nunca versionar no Git)

- **Arquivo esperado:** `RBAC43EMD05.pdf`
- **Tamanho esperado:** `349196` bytes
- **SHA-256 esperado:** `5026d2da87fc3f55c61a6b5a20dd1f50d52a1367a8550897f3864074e1a46472`

### 2.2. Ingestão idempotente via `pdfplumber` (offline, zero chamadas Gemini)

```powershell
Set-Location "C:\Projetos\projeto-fundamentos-de-IA"
$pdfPath = "C:\Projetos\projeto-fundamentos-de-IA\data\raw_pdfs\RBAC43EMD05.pdf"

docker cp $pdfPath rbac-bot:/tmp/RBAC43EMD05.pdf
docker compose exec -T rbac-bot python scripts/ingest_regulation.py --document-id anac_rbac_43 --pdf-path /tmp/RBAC43EMD05.pdf
docker compose exec -T rbac-bot python -c "from pathlib import Path; Path('/tmp/RBAC43EMD05.pdf').unlink(missing_ok=True)"
docker compose exec -T rbac-bot python scripts/inspect_document.py --document-id anac_rbac_43
docker compose exec -T rbac-bot python scripts/inspect_document.py --document-id anac_rbac_43 --section 43.1
```

### 2.3. Revisão humana explícita (nunca existe aprovação cega em massa)

```powershell
# 1. Inspecionar páginas/seção extraídas e conferir com o PDF oficial ANAC
docker compose exec -T rbac-bot python scripts/validate_extraction.py --document-id anac_rbac_43 --section 43.1
docker compose exec -T rbac-bot python scripts/validate_extraction.py --document-id anac_rbac_43 --section 43.3

# 2. Aprovar explicitamente somente após conferência real com o PDF oficial
docker compose exec -T rbac-bot python scripts/validate_extraction.py --document-id anac_rbac_43 --section 43.1 --decision approve --reviewer "Arthur Fava" --comments "Conferido com PDF oficial ANAC RBAC 43 Emenda 05" --confirm
docker compose exec -T rbac-bot python scripts/validate_extraction.py --document-id anac_rbac_43 --section 43.3 --decision approve --reviewer "Arthur Fava" --comments "Conferido com PDF oficial ANAC RBAC 43 Emenda 05" --confirm
```

### 2.4. Indexação controlada (SQLite FTS5 + ChromaDB `gemini-embedding-2` 768d Free Tier)

> **Atenção:** este passo consome cota Free Tier da `GEMINI_API_KEY` local (`GEMINI_ALLOW_PAID_MODELS=false`, sem Cloud Billing).

```powershell
docker compose exec -T rbac-bot python scripts/index_document.py --document-id anac_rbac_43
docker compose exec -T rbac-bot python scripts/index_document.py --status
```

### 2.5. Avaliação real e calibração reproduzível de `max_vector_distance`

```powershell
# Somente lexical/estrutural (custo zero)
docker compose exec -T rbac-bot python scripts/evaluate_rag.py --corpus official-approved --embedding-backend deterministic --per-query

# Avaliação completa com gemini-embedding-2 real + recomendação de calibração (consome cota Free Tier)
docker compose exec -T rbac-bot python scripts/evaluate_rag.py --corpus official-approved --embedding-backend gemini-embedding-2 --per-query
```

O comando **não** altera `config/settings.yaml` automaticamente. Caso a recomendação comprove ganho de Precision/MRR preservando `100%` de recusa segura e `0` violações de governança no RBAC 43 real, registre a justificativa em ADR/PR antes de alterar `retrieval.max_vector_distance` (atualmente mantido em **`0.85` provisório**).
