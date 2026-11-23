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

---

## 3. Escopo Normativo Estrito e Controles Anti-Alucinação na Operação Pública

Embora a interface Telegram opere em **modo público** para consultas em conversas privadas (`public_access: true`, `default_role: USER`), o conhecimento respondido pelo motor RAG permanece estritamente limitado ao corpus normativo aprovado e indexado:

1. **Recuperação restrita a páginas aprovadas (`retrieval.telegram_only_approved_literal: true`):**
   - Toda consulta recebida pelo `TelegramCommandRouter` envia `SearchFilter(only_current_version=True, only_approved_literal=True, include_preliminary_native=False)` ao `HybridNormativeRetriever`.
   - Os três sinais de busca (**estrutural exato**, **lexical FTS5** e **vetorial ChromaDB**) excluem páginas pendentes de revisão, preliminares não revisadas, OCR não revisado e páginas reprovadas.
2. **Proibição de busca externa e de preenchimento por conhecimento geral:**
   - Não existe fallback de busca na internet nem consulta externa.
   - Sem evidência aprovada suficiente (incluindo perguntas fora do escopo documental), o `NormativeChatService` devolve recusa fundamentada **antes** de chamar o Gemini (`llm.calls == 0`).
3. **Separação estrita SSOT vs. explicação IA e validação determinística:**
   - O bloco `TEXTO NORMATIVO (LITERAL VALIDADO — SQLite SSOT)` é entregue diretamente do SQLite sem reescrita pelo modelo.
   - O `CitationValidator` audita `[EV-N]`, seções, artigos e páginas produzidos na explicação IA e descarta referências não confirmadas nas evidências recuperadas.
   - O `PromptGuard` trata perguntas e evidências como dados não confiáveis (detectando *prompt injection*, exfiltração de prompts/variáveis/credenciais, escalada de privilégio e quebra de escopo), e o `secret_redactor` sanitiza saídas antes da entrega.
4. **Rastreabilidade e transparência técnica:**
   - Cada evidência preserva documento, autoridade, emenda, seção e páginas (PDF/impressa).
   - Esses controles verificáveis (cobertos por `tests/evals/test_regulatory_fidelity_regression.py`) minimizam e detectam respostas sem fundamento — sem reivindicar garantia matemática absoluta contra alucinação, privilegiando sempre a recusa fundamentada quando houver dúvida ou ausência de evidência.
