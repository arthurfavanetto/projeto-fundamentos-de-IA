# Registro de Decisões Arquiteturais (ADRs)

Este documento consolida as Decisões Arquiteturais (*Architecture Decision Records*) do **RBAC AI Assistant** (Domínio: Regulamentação Aeronáutica Brasileira — ANAC).

---

## ADR-01 — Separação entre Fonte Normativa Primária (PDF), SSOT Operacional (SQLite) e Índice Vetorial (ChromaDB)

- **Status:** Aprovada (atualizada nas Etapas 2 e 3)
- **Contexto:** Bancos vetoriais otimizam busca por similaridade semântica, mas fragmentam documentos normativos em chunks. Em regulamentação aeronáutica, omitir um parágrafo de exceção ou alterar pontuação compromete a validade técnica e jurídica da orientação.
- **Decisão:** Preservar o **PDF oficial da ANAC** intacto no volume `regulatory_documents` como fonte normativa primária, utilizar o **SQLite** (`/app/data/documents/catalog.sqlite3`) como fonte de verdade operacional dos dados estruturados e revisados (mantendo `original_text` separado de `normalized_text`) e restringir o **ChromaDB** ao papel de índice vetorial derivado e reconstruível. (Detalhes consolidados na Seção 2 do `README.md`.)
- **Consequências:**
  - Transcrições oficiais (`TEXTO NORMATIVO`) são lidas exclusivamente do campo `original_text` no SQLite em páginas aprovadas, sem jamais passar por geração ou reescrita do LLM.
  - Tanto o índice lexical FTS5 quanto a coleção vetorial do ChromaDB podem ser reconstruídos integralmente a partir do SQLite sem perda de revisões humanas.

---

## ADR-02 — Arquitetura Hexagonal (*Ports and Adapters*) com *src-layout*

- **Status:** Aprovada
- **Contexto:** O MVP inicia focado no RBAC 43, usando Telegram, ChromaDB e Google Gemini, mas o projeto precisará escalar para outros regulamentos (RBAC 145, RBAC 65, IS) e eventualmente suportar novas interfaces (Web/API) ou novos modelos de IA.
- **Decisão:** Isolar `domain/` e `application/` de `infrastructure/` e `interfaces/`, estruturando o código em `src/rbac_assistant/` com contratos explícitos em `domain/ports/`.
- **Consequências:**
  - Núcleo de domínio 100% testável sem dependências de APIs externas ou bancos.
  - Pequeno custo inicial de organização de camadas amplamente compensado por manutenibilidade e testabilidade.

---

## ADR-03 — Telegram Long Polling no MVP em vez de Webhooks

- **Status:** Aprovada
- **Contexto:** O ambiente inicial de operação é um PC Central Windows doméstico/corporativo atrás de roteador/NAT, com possível migração futura para VPS Linux.
- **Decisão:** Adotar **HTTPS Long Polling** (`getUpdates`) originado de dentro do contêiner `rbac-bot`.
- **Consequências:**
  - Dispensa IP fixo, abertura de portas no roteador, certificados TLS públicos ou túneis reversos.
  - Mesma configuração funciona sem alterações no Windows (Docker Desktop) e em VPS Linux.
  - Requer garantir instância única do bot por token para evitar conflito (`409 Conflict`).

---

## ADR-04 — Recuperação Híbrida (Determinística Prioritária + Semântica)

- **Status:** Aprovada
- **Contexto:** Em consultas regulatórias, buscar `"43.3"` apenas por distância de cosseno frequentemente recupera seções lexicalmente próximas como `"43.13"` ou `"43.5"`.
- **Decisão:** Priorizar a resolução determinística (`SectionResolver` + lookup exato no SSOT por `regulation_number` + `amendment_number` + `section_number`) sempre que houver identificador normativo explícito, recorrendo à busca semântica vetorial no ChromaDB (com reconstrução integral da seção no SSOT) para perguntas temáticas/abertas.
- **Consequências:**
  - Recall de 100% em consultas diretas por seção.
  - Proteção contra mistura silenciosa entre emendas vigentes e históricas via filtro `is_current_amendment=True` por padrão.

---

## ADR-05 — Topologia Mínima de Dois Contêineres (`rbac-bot` + `chromadb`)

- **Status:** Aprovada
- **Contexto:** O sistema deve operar de forma contínua e econômica em recursos no PC Central Windows (Docker Desktop) e ser facilmente migrado para uma VPS Linux.
- **Decisão:** Limitar a stack Docker aos dois serviços essenciais (`rbac-bot` e `chromadb`), utilizando persistência atômica em volumes nomeados (`regulatory_documents`, `chroma_data`, `audit_logs`) e logs estruturados em JSONL.
- **Consequências:**
  - Elimina overhead de memória e manutenção de serviços auxiliares (PostgreSQL, Redis, Nginx) no MVP.
  - Simplifica backup, restauração e portabilidade entre Windows e Linux.

---

## ADR-06 — Separação de Templates Versionados e Segredos/IDs em Repositório Público

- **Status:** Aprovada
- **Contexto:** O repositório GitHub do projeto é público; tokens de API e Telegram User IDs reais não devem ser expostos no histórico Git.
- **Decisão:** Versionar apenas `.env.example`, `config/access_control.example.yaml` e `config/settings.yaml` (não-sensível), mantendo `.env`, `config/access_control.yaml`, `data/` e `*.pdf` bloqueados no `.gitignore`.
- **Consequências:**
  - Operadores copiam os templates localmente no host antes do deploy.
  - Reduz drasticamente o risco de vazamento acidental de credenciais ou dados de usuários.

---

## ADR-07 — SQLite (SSOT Operacional + FTS5) + ChromaDB 1.5.9 com Fusão RRF (Etapas 2 e 3)

- **Status:** Aprovada
- **Contexto:** O armazenamento operacional da aplicação consolidou-se em SQLite 3 (`/app/data/documents/catalog.sqlite3`, modo WAL, `FOREIGN KEYs ON`, `synchronous = FULL`, transações ACID) no volume `regulatory_documents`, enquanto o PDF oficial da ANAC permanece como fonte normativa primária. Na Etapa 3, consultas regulatórias precisam combinar referências estruturais (`RBAC 43.3`, `Art. 1º`), termos exatos com/sem acento (`manutencao` vs. `manutenção`) e busca semântica aberta.
- **Decisão:**
  - Adotar índice lexical derivado **SQLite FTS5** (`normative_chunks_fts`, tokenizador `unicode61 remove_diacritics 2`) via migração incremental `v002_fts5_and_chunks.sql`, preservando `original_text` separado de `normalized_text` e oferecendo reconstrução completa (`rebuild_fts_index()`).
  - Combinar busca estrutural exata (`section_resolver`), busca lexical FTS5 (BM25) e busca vetorial no ChromaDB 1.5.9 via **Reciprocal Rank Fusion (RRF)**, com limiar inicial/provisório `max_vector_distance=0.85` (distância de cosseno `hnsw:space: cosine`, pendente de calibração por benchmark real com o RBAC 43).
  - Hidratar todas as evidências ao vivo a partir do SQLite aplicando `ValidationService` (`review_on_demand`), bloqueando páginas OCR pendentes ou reprovadas.
- **Consequências:**
  - Nenhuma dependência externa adicional para busca lexical (FTS5 é nativo no Python 3.12 / SQLite 3.53).
  - Aprovações ou reprovações de páginas refletem imediatamente nas consultas sem necessidade de reindexar vetores.

---

## ADR-08 — SDK Oficial `google-genai`, Gemini 4 Hydrogen (Free Tier Standard) e Gemini Embedding 2 (768d)

- **Status:** Aprovada (revisão técnica final da Etapa 3)
- **Contexto:** É obrigatória a separação entre (1) o ambiente de desenvolvimento no Antigravity (assinatura paga Google AI Pro, nunca via API key) e (2) o runtime da aplicação Python via **Gemini Developer API Free Tier Standard** (`google-genai==3.6.1`). Na tabela oficial da Gemini API, `gemini-4-hydrogen` (alternativa gratuita: `gemini-3.8-flash`) e `gemini-embedding-2` (modalidade texto, 8.192 tokens, MRL 128–3072) operam no Free Tier Standard sem exigir Google Cloud Billing, ao passo que `gemini-4-neon`, `gemini-4-carbon` e `gemini-4-argon` são exclusivos do tier pago. Além disso, `gemini-embedding-2` possui espaço vetorial incompatível com `gemini-embedding-001`, não utiliza `task_type`, exige instruções assimétricas no texto e agrega múltiplos inputs se não encapsulados em objetos `types.Content` individuais.
- **Decisão:**
  - Definir `GEMINI_GENERATION_MODEL=gemini-4-hydrogen` como padrão de geração (Free Tier Standard), mantendo `gemini-3.8-flash` como alternativa gratuita configurável, **sem fallback silencioso** entre modelos, sem Batch API e sem ferramentas pagas (Google Search/Maps grounding).
  - Implementar allowlist de modelos gratuitos em `AppSettings` (`GEMINI_ALLOW_PAID_MODELS=false` por padrão), rejeitando modelos pagos no carregamento de configuração a menos que explicitamente autorizados, documentando claramente que o código não substitui a manutenção da conta sem faturamento no Google Cloud e alertando sobre a política de uso de dados do Free Tier (apenas normas públicas da ANAC).
  - Adotar `GEMINI_EMBEDDING_MODEL=gemini-embedding-2` com `GEMINI_EMBEDDING_DIMENSIONS=768` (dimensão MRL recomendada e auto-normalizada, compatível com distância de cosseno no ChromaDB), com convenção oficial assimétrica aplicada apenas em memória (`task: search result | query: {consulta}` e `title: {título|none} | text: {conteúdo}`) e 1 objeto `types.Content` por chunk.
  - Versionar a coleção de produção como **`regulatory_chunks_ge2_768_v1`**, registrando modelo, dimensão, versão da estratégia (`ge2_asymmetric_v1` — migração `v003_index_strategy_metadata.sql`) e data de criação para bloquear qualquer mistura de espaços vetoriais (`EmbeddingCompatibilityError`).
  - Bloquear integralmente o envio de páginas não aprovadas (`native_quality_checked`, OCR pendente ou reprovadas) ao modelo generativo e validar determinísticamente no servidor todas as citações `[EV-N]` e seções geradas via `CitationValidator`.
- **Consequências:**
  - Elimina risco de cobrança acidental por seleção de modelos pagos, agregação silenciosa de embeddings ou mistura de vetores incompatíveis.
  - Garante rastreabilidade e precisão de citações (`[EV-N]`), preserva o texto literal exclusivamente no SQLite e permite migração futura controlada via rebuild vetorial.

