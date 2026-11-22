# Estrutura Normativa ANAC e Estratégia de Extração

Este documento especifica a gramática estrutural dos Regulamentos Brasileiros da Aviação Civil (RBAC) publicados pela ANAC, o modelo de metadados obrigatórios e as regras de preservação literal de texto.

---

## 1. Hierarquia Estrutural dos RBACs da ANAC

O parser estrutural (`rag/ingestion/structure_parser.py`) deve reconhecer e preservar a hierarquia normativa oficial da ANAC:

1. **Preâmbulo / Capa / Controle de Emendas / Sumário**
2. **Subparte** — ex.: `Subparte A — Geral`, `Subparte B`
3. **Seção** — ex.: `43.1`, `43.3`, `43.9`, `43.13`
4. **Parágrafo** — ex.: `(a)`, `(b)`, `(c)`
5. **Inciso** — ex.: `(1)`, `(2)`, `(3)`
6. **Alínea** — ex.: `(i)`, `(ii)`, `(iii)`
7. **Subitem** — ex.: `(A)`, `(B)`, `(C)`
8. **Apêndices e Tabelas Normativas** — ex.: `Apêndice A`, `Apêndice B`

> **Regra de Integridade:** A pontuação original (`;`, `:`, `.`, `—`, parênteses), termos modais (*"deve"*, *"pode"*, *"exceto"*, *"somente"*) e a numeração de dispositivos não podem ser alterados.

---

## 2. Estratégia de Extração de PDF

### 2.1. Extração Nativa Prioritária
- Utiliza `PyMuPDF` (`fitz`) e `pdfplumber` com leitura por coordenadas espaciais (*bounding boxes*).
- Identifica e extrai a numeração de **página impressa oficial da ANAC** (`printed_page_number`) do cabeçalho/rodapé antes de remover elementos repetitivos de diagramação.
- Preserva simultaneamente a **página física do PDF** (`pdf_page_number`, 1-based).
- Preserva tabelas normativas em formato tabular estruturado sem embaralhar colunas.

### 2.2. OCR Condicional por Página e Política `block_until_reviewed`
- O OCR (`Tesseract` com idioma `por`) só é acionado para **páginas individuais** cuja camada de texto nativo fique abaixo do limiar configurado (`min_native_chars_per_page`).
- Cada página carrega metadados explícitos de `extraction_method` (`"native_pdf"` ou `"ocr_fallback"`) e `validation_status` (`"validated"`, `"blocked_unreviewed"`, `"pending_review"` ou `"failed"`).
- **Política aprovada (`block_until_reviewed`):** toda página que depender de OCR fica **bloqueada individualmente** para transcrição normativa literal até ser revisada manualmente e aprovada.
- Páginas extraídas diretamente do PDF (`native_pdf`) também passam por verificações de integridade e preservação textual antes de receber o status `"validated"`.
- Quando uma seção abranger qualquer página não validada, o sistema sinaliza explicitamente que **não pode garantir a transcrição literal completa** daquela seção.

### 2.3. Normalização Não-Destrutiva
- Remove apenas quebras de linha artificiais no meio de uma frase decorrentes da largura da coluna do PDF (*unhyphenation* conservadora).
- Preserva integralmente quebras estruturais de parágrafos `(a)`, incisos `(1)`, alíneas `(i)` e subitens `(A)`.

---

## 3. Esquema dos 18 Metadados Obrigatórios por Chunk

Todos os chunks persistidos no ChromaDB e referenciados no índice canônico devem carregar os 18 campos abaixo:

| # | Campo | Tipo | Descrição |
| :-: | :--- | :--- | :--- |
| 1 | `document_id` | `str` | Identificador único determinístico (ex.: `rbac_43_emd_xx`) |
| 2 | `regulation_name` | `str` | Nome completo (ex.: `"Regulamento Brasileiro da Aviação Civil nº 43"`) |
| 3 | `regulation_number` | `str` | Código curto (ex.: `"RBAC 43"`) |
| 4 | `amendment_number` | `str` | Número da emenda oficial verificada na ANAC |
| 5 | `document_title` | `str` | Título oficial do regulamento |
| 6 | `publication_date` | `str` | Data ISO-8601 de publicação (DOU / ANAC) |
| 7 | `effective_date` | `str` | Data ISO-8601 de início de vigência |
| 8 | `official_url` | `str` | URL oficial verificada no portal da ANAC |
| 9 | `pdf_filename` | `str` | Nome do arquivo PDF original no volume |
| 10 | `pdf_page_number` | `int` | Número físico da página no PDF (1-based) |
| 11 | `printed_page_number` | `str` | Numeração impressa oficial da página ANAC |
| 12 | `section_number` | `str` | Identificador da seção (ex.: `"43.3"`, `"Apêndice A"`) |
| 13 | `subsection_number` | `str` | Subseção ou título da seção |
| 14 | `paragraph_number` | `str` | Parágrafo (ex.: `"(a)"` ou `""`) |
| 15 | `item_number` | `str` | Inciso/alínea (ex.: `"(1)(i)"` ou `""`) |
| 16 | `content_hash` | `str` | Hash SHA-256 do conteúdo textual do chunk |
| 17 | `extraction_method` | `str` | `"native_pdf"` ou `"ocr_fallback"` |
| 18 | `extraction_validation_status` | `str` | `"validated"`, `"requires_review"` ou `"failed"` |

---

## 4. Chunking Estrutural Hierárquico

- **Seções dentro da janela alvo (até `max_chunk_tokens`, padrão 1.200 — folga ampla abaixo dos 8.192 tokens do `gemini-embedding-2`):** mantidas íntegras em um único chunk.
- **Seções extensas (ex.: `43.3`, `43.13`, Apêndices):** divididas nas fronteiras naturais de linhas/parágrafos sem quebrar a rastreabilidade de página e seção.
- **Representação auxiliar para busca e embeddings:** o prefixo assimétrico do `gemini-embedding-2` (`title: ... | text: ...`) é aplicado apenas de forma efêmera na vetorização; o banco **SQLite** (`/app/data/documents/catalog.sqlite3`) preserva `original_text` puro separado de `normalized_text` (ver Seção 2 do [`README.md`](../../README.md#2-arquitetura-de-dados-sqlite-e-fontes-de-verdade)).
- **Reconstrução integral:** ao recuperar qualquer chunk via busca híbrida (FTS5 + ChromaDB + RRF), o `HybridNormativeRetriever` hidrata a seção completa a partir do SQLite e valida ao vivo todas as páginas abrangidas.
