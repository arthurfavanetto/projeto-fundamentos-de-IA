# RBAC AI Assistant (ANAC)

Assistente normativo de Inteligência Artificial especializado em **Regulamentos Brasileiros da Aviação Civil (RBAC)** e demais atos normativos publicados pela **Agência Nacional de Aviação Civil (ANAC)**, iniciado com foco no **RBAC 43** (*Manutenção, Manutenção Preventiva, Reconstrução e Alteração*), projetado em **Arquitetura Hexagonal (*Ports and Adapters*)** com Python 3.12+, `uv` e execução em contêineres **Docker non-root** (UID/GID `10001`).

> **Nota terminológica:** neste projeto, `RBAC` significa exclusivamente **Regulamento Brasileiro da Aviação Civil (ANAC)** — nunca *Role-Based Access Control*. O controle de acesso operacional do bot (Etapa 4) utiliza `access_control` e `operator_roles` (`ADMIN` / `USER`).

---

## 1. Visão Geral

O sistema permite ingerir documentos regulatórios da ANAC, preservar o texto oficial integralmente com controle de revisão por página (`review_on_demand`), executar busca híbrida (estrutural + SQLite FTS5 + ChromaDB com fusão **RRF**) e gerar explicações técnicas fundamentadas via **Google Gemini** (SDK oficial `google-genai==3.6.1`).

Toda resposta separa rigorosamente **4 blocos conceituais**:

1. **TEXTO NORMATIVO:** transcrição literal obtida diretamente do SQLite, exibida como validada apenas quando todas as páginas abrangidas estão aprovadas manualmente.
2. **EXPLICAÇÃO:** interpretação técnica gerada por IA (Gemini) estritamente sobre as evidências recuperadas.
3. **REFERÊNCIA:** regulamento, emenda, vigência, seção/dispositivo, página física do PDF, página impressa da ANAC e estado de validação.
4. **OBSERVAÇÃO:** avisos de revisão preliminar, bloqueio de páginas OCR/reprovadas, vigência ou degradação controlada.

---

## 2. Arquitetura de Dados, SQLite e Fontes de Verdade

A arquitetura separa de forma inequívoca a **fonte normativa primária**, a **fonte de verdade operacional** e os **índices derivados reconstruíveis**.

### 2.1. PDF Original — Fonte Normativa Primária

O arquivo PDF oficial publicado pela ANAC é a **fonte normativa documental primária** do sistema. Ele é preservado sem alterações no volume persistente (`regulatory_documents`, subdiretório `/app/data/documents/raw_pdfs/`), identificado por hash criptográfico **SHA-256** e acompanhado de metadados de autoridade, tipo documental, regulamento, versão/emenda, datas de publicação/vigência e URL oficial. O PDF original jamais é substituído pela saída do Gemini ou por textos normalizados.

### 2.2. SQLite — Fonte de Verdade Operacional

O banco **SQLite** localizado em caminho único e persistente — `/app/data/documents/catalog.sqlite3` (dentro do volume Docker `regulatory_documents` montado em `/app/data/documents`) — é a **fonte de verdade operacional** da aplicação e dos dados estruturados extraídos e revisados. Ele armazena com integridade transacional (`PRAGMA foreign_keys = ON`, `journal_mode = WAL`, `synchronous = FULL`, `busy_timeout`, migrações incrementais versionadas em `schema_migrations` e backup online consistente):

- catálogo multidocumento e identificação de regulamentos/tipos documentais;
- versões e emendas (`document_versions`) e histórico de extrações (`extraction_runs`);
- páginas extraídas (`regulatory_pages`) com paginação física e impressa;
- texto original extraído (`original_text`) e texto normalizado separado (`normalized_text`);
- estrutura normativa hierárquica (`normative_units`, tabelas e referências cruzadas);
- estados de revisão por página e trilha auditável de aprovações/reprovações (`validation_records`);
- chunks estruturais (`normative_chunks`) e registro de versões indexadas (`vector_index_registry`) necessários para reconstruir os índices derivados.

O SQLite é a fonte de verdade operacional da aplicação; **não substitui o PDF original da ANAC como fonte normativa primária**.

### 2.3. Texto Original e Texto Normalizado

O sistema mantém separação estrita entre `original_text` e `normalized_text`:

- `original_text` preserva a extração tal como obtida do PDF, mantendo acentuação em português, pontuação, ordem e redação original.
- `normalized_text` (e a representação auxiliar de busca `search_text`) facilita pesquisa e indexação sem jamais modificar ou sobrescrever `original_text`.
- Toda transcrição normativa literal entregue ao usuário é montada exclusivamente a partir de `original_text` recuperado do SQLite em páginas com estado `approved_for_literal_transcription`.
- Prefixos de instrução usados na geração de embeddings (ex.: `title: ... | text: ...`) são construídos apenas em memória no momento da vetorização e nunca são gravados em `original_text`.

### 2.4. SQLite FTS5 — Índice Textual

O **SQLite FTS5** (tabela virtual `normative_chunks_fts`, criada na migração incremental `v002_fts5_and_chunks.sql` com tokenizador `unicode61 remove_diacritics 2`) é o **índice derivado de pesquisa textual** (BM25, insensível a acentos e caixa). Ele não é fonte normativa primária, não substitui `original_text` e pode ser integralmente reconstruído a qualquer momento a partir da tabela `normative_chunks` do SQLite via `rebuild_fts_index()`.

### 2.5. ChromaDB — Índice Vetorial

O **ChromaDB 1.5.9** (executado em contêiner dedicado non-root com porta `8000` isolada exclusivamente na rede interna `rbac_internal_net`) é unicamente o mecanismo de armazenamento e busca por similaridade vetorial (métrica de distância de cosseno `hnsw:space: cosine`). Ele armazena vetores e metadados de rastreabilidade ligados ao SQLite, **não é fonte de verdade dos documentos** e seus índices são totalmente reconstruíveis a partir dos dados persistidos no SQLite.

A coleção de produção versionada para o `gemini-embedding-2` (768 dimensões, estratégia assimétrica `ge2_asymmetric_v1`) é **`regulatory_chunks_ge2_768_v1`**. O sistema registra modelo, dimensionalidade, versão da estratégia de instrução e data de criação na coleção e em `vector_index_registry`, impedindo mistura silenciosa de espaços vetoriais incompatíveis (como `gemini-embedding-001` e `gemini-embedding-2`, mesmo quando ambos usam 768 dimensões).

### 2.6. Gemini — Embeddings e Explicações

A API oficial do **Google Gemini** (`google-genai==3.6.1`) cumpre dois papéis estritamente delimitados:

- **Embeddings (`gemini-embedding-2`, 768d obrigatório):** vetorização de chunks e consultas utilizando a convenção oficial assimétrica (`title: {título} | text: {conteúdo}` para documentos e `task: search result | query: {consulta}` para buscas, sem `task_type` e com objetos `Content` individuais por chunk para impedir agregação). Modelos legados como `gemini-embedding-001` não são aceitos em runtime.
- **Explicações (`gemini-4-hydrogen`, único modelo de geração permitido no Free Tier Standard):** elaboração de explicações técnicas fundamentadas exclusivamente nas evidências **aprovadas** recuperadas do SQLite, protegidas pelo `PromptGuard` e auditadas server-side pelo `CitationValidator` (`[EV-N]`). Páginas não aprovadas (`native_quality_checked`, OCR pendente ou reprovadas) **não são enviadas ao Gemini**.

O Gemini **não é fonte normativa primária** e **não pode criar, corrigir, completar ou substituir redações normativas**.

### 2.7. Persistência e Reconstrução

A stack Docker (`docker-compose.yml`) opera com dois contêineres non-root (UID/GID `10001`) e três volumes nomeados persistentes:

- `regulatory_documents` → `/app/data/documents` (PDFs originais imutáveis, banco `/app/data/documents/catalog.sqlite3`, snapshots extraídos e trilhas de revisão);
- `chroma_data` → `/data` no contêiner `chromadb` (índice vetorial derivado);
- `audit_logs` → `/app/logs` (logs estruturados).

Caso o índice lexical ou o índice vetorial precise ser recriado (ex.: rebuild administrativo ou migração controlada de versão), ambos são reconstruídos diretamente a partir do SQLite sem reingerir PDFs e preservando 100% das aprovações humanas de revisão:

- **Reconstrução FTS5:** `SqliteDocumentRepository.rebuild_fts_index()`.
- **Reconstrução vetorial ChromaDB:** `python scripts/index_document.py --rebuild-collection --confirm`.

---

## 3. Modelos Gemini (Desenvolvimento vs. Runtime Free Tier), Privacidade e Configuração Inicial

Arquivos com credenciais, Telegram User IDs ou PDFs oficiais **nunca são versionados no Git** (`.gitignore`).

### 3.1. Separação obrigatória entre ambiente de desenvolvimento e runtime da aplicação

- **Desenvolvimento (Antigravity):** utiliza a assinatura pessoal **Google AI Pro** diretamente no seletor de modelos do Antigravity — nunca via chave de API do projeto.
- **Execução da aplicação (bot Python):** utiliza a **Gemini Developer API** via chave `GEMINI_API_KEY` sob **política única de modelos em Free Tier Standard**:
  - **Geração (modelo único permitido):** `GEMINI_GENERATION_MODEL=gemini-4-hydrogen` (Free Tier Standard; qualquer outro modelo — incluindo `gemini-3.5-flash-lite`, `gemini-3.8-flash` ou modelos pagos — é rejeitado explicitamente na validação de configuração e no adaptador, **sem fallback silencioso**).
  - **Embeddings (modelo único e dimensão obrigatória):** `GEMINI_EMBEDDING_MODEL=gemini-embedding-2` com `GEMINI_EMBEDDING_DIMENSIONS=768` (Free Tier Standard, modalidade somente texto; `gemini-embedding-001` e outras dimensões são rejeitados em runtime, **sem fallback silencioso**).
  - **Política de custos (`GEMINI_ALLOW_PAID_MODELS=false`):** obrigatório pela política do projeto; impede modelos exclusivos do tier pago (ex.: `gemini-4-neon`, `gemini-4-carbon`, `gemini-4-argon`), sem fallback pago, sem Batch API e sem Google Search/Maps grounding.
  - **Aviso importante sobre cobrança:** a configuração local no código **não garante sozinha ausência absoluta de cobrança** — isso também depende do projeto Google associado à `GEMINI_API_KEY` estar no Free Tier **sem Google Cloud Billing vinculado**.

### 3.2. Limites e privacidade no Gemini API Free Tier

- O **Free Tier** possui limites de taxa (RPM/TPM/RPD); o cliente aplica retry com backoff exponencial em HTTP 429 e degradação graciosa preservando o texto normativo do SQLite.
- Conforme os termos do **Gemini API Free Tier**, prompts e respostas podem ser usados pelo Google para aprimoramento de produtos. Portanto, no Free Tier envie **exclusivamente documentos regulatórios públicos oficiais da ANAC** — nunca dados pessoais, corporativos sigilosos ou credenciais.

### 3.3. Configuração do `.env` local

1. Copie `.env.example` para `.env` localmente e configure:

```env
GEMINI_API_KEY=
GEMINI_GENERATION_MODEL=gemini-4-hydrogen
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMENSIONS=768
GEMINI_ALLOW_PAID_MODELS=false
```

2. **Separação metodológica das avaliações RAG e threshold provisório:**
   - **Avaliação sintética determinística (CI / offline):** `scripts/evaluate_rag.py` usa 40 consultas sobre corpus sintético fictício com embedder determinístico local (`eval-deterministic-embed-v1`, 768d) — **não** representa qualidade semântica real do `gemini-embedding-2` nem respostas reais do `gemini-4-hydrogen`.
   - **Benchmark manual opcional com API real:** `scripts/evaluate_rag.py --embedding-backend compare` compara o embedder sintético com `gemini-embedding-2` (768d, Free Tier) ainda sobre o corpus sintético (fora do CI, consome cota da `GEMINI_API_KEY` local).
   - **Validação futura com PDF real do RBAC 43:** o limiar `retrieval.max_vector_distance: 0.85` (distância de cosseno no ChromaDB) é **provisório** e deverá ser recalibrado após a ingestão real do PDF oficial do RBAC 43 com `gemini-embedding-2` (768d).
3. Para ingestão, validação, indexação, diagnóstico (`--debug`), benchmark sintético (`--embedding-backend compare`) e consulta passo a passo, consulte o **[Runbook Operacional](docs/operations/03_ingestion_runbook.md)**.

---

## 4. Estrutura do Repositório

```text
.
├── config/
│   ├── prompts/                   # Templates versionados do motor RAG (Markdown)
│   ├── access_control.example.yaml
│   ├── regulations_catalog.yaml   # Manifesto declarativo de metadados ANAC
│   └── settings.yaml              # Configurações operacionais não-sensíveis
├── docker/
│   └── chromadb.Dockerfile        # ChromaDB 1.5.9 oficial rodando como UID/GID 10001
├── docs/
│   ├── architecture/              # Arquitetura, ADRs (ADR-01 a ADR-08) e gramática ANAC
│   ├── evaluations/               # Protocolo de fidelidade normativa
│   ├── legacy/                    # Notas históricas arquivadas (somente referência histórica)
│   └── operations/                # Guias Windows, VPS Linux e Runbook operacional
├── scripts/
│   ├── ingest_regulation.py       # Ingestão genérica de PDFs regulatórios para o SQLite
│   ├── inspect_document.py        # Inspeção de catálogo, versões, páginas e seções
│   ├── validate_extraction.py     # Revisão humana explícita por página/intervalo/seção
│   ├── index_document.py          # Indexação (FTS5 + ChromaDB), diagnóstico e rebuild
│   ├── query_assistant.py         # Consulta CLI ao motor RAG híbrido (incl. --debug e --no-llm)
│   ├── evaluate_rag.py            # Benchmark controlado (40 consultas, --per-query e --embedding-backend)
│   ├── verify_chroma_persistence.py
│   └── verify_sqlite_persistence.py
├── src/rbac_assistant/
│   ├── config/                    # Pydantic Settings (com allowlist Free Tier) e structlog
│   ├── domain/                    # Entidades puras e Ports (zero dependências externas)
│   ├── application/               # DTOs, Serviços (Catalog, Ingestion, Indexing, Chat) e Use Cases
│   ├── security/                  # PromptGuard e secret_redactor
│   ├── rag/                       # Chunker, SectionResolver, HybridRetriever, Reranker, CitationValidator, Eval
│   ├── infrastructure/            # SQLite+FTS5, ChromaDB 1.5.9, Google GenAI e pdfplumber
│   └── interfaces/telegram/       # (Etapa 4) Interface Telegram Long Polling
├── tests/                         # Testes unitários, integração e evals sintéticos controlados
├── Dockerfile                     # Imagem multi-stage Python 3.12 + uv non-root (UID 10001)
├── docker-compose.yml             # Stack rbac-bot + chromadb + redes e volumes persistentes
└── pyproject.toml
```

---

## 5. Documentação Complementar

- [Runbook Operacional — Ingestão, Validação, Indexação e Consulta](docs/operations/03_ingestion_runbook.md)
- [Arquitetura do Sistema](docs/architecture/01_system_architecture.md)
- [Decisões Arquiteturais (ADRs)](docs/architecture/02_adr_decisions.md)
- [Estrutura Normativa ANAC e Metadados](docs/architecture/03_anac_normative_structure.md)
- [Setup Local — PC Central Windows (Docker Desktop)](docs/operations/01_local_windows_setup.md)
- [Migração para VPS Linux](docs/operations/02_vps_linux_migration.md)
- [Protocolo de Fidelidade Normativa](docs/evaluations/normative_fidelity_protocol.md)

---

## 6. Estado das Etapas do Projeto

| Etapa | Escopo | Status |
| :--- | :--- | :--- |
| **Etapa 0** | Arquitetura, documentação base, templates e arquivamento de legado | Concluída e integrada (`main`) |
| **Etapa 1** | Fundação Python 3.12 (`uv`), arquitetura hexagonal, Docker non-root e ChromaDB 1.5.9 | Concluída e integrada (`main`) |
| **Etapa 2** | Motor multidocumento de ingestão documental (SQLite WAL/ACID, `pdfplumber`, validação híbrida `review_on_demand`) | Concluída e integrada (`main`) |
| **Etapa 3** | Integração Gemini Free Tier (`gemini-4-hydrogen` + `gemini-embedding-2` 768d), SQLite FTS5, ChromaDB versionado, Motor RAG Híbrido (RRF), `CitationValidator`, `--debug` e avaliação controlada | **Implementada na branch `feat/etapa-3-gemini-hybrid-rag` (PR #7) — aguardando revisão e merge** |
| **Etapa 4** | Interface Telegram (Long Polling), controle de acesso (`ADMIN`/`USER`), comandos e `MessageSplitter` | Futura |
| **Etapa 5** | Evals avançados de fidelidade com dataset real (RBAC 43 ingerido), recalibração de thresholds e automação de backup | Futura |