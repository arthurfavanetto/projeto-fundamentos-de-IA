# Registro de Decisões Arquiteturais (ADRs)

Este documento consolida as Decisões Arquiteturais (*Architecture Decision Records*) do **RBAC AI Assistant** (Domínio: Regulamentação Aeronáutica Brasileira — ANAC).

---

## ADR-01 — Separação entre Armazenamento Canônico (SSOT) e Banco Vetorial (ChromaDB)

- **Status:** Aprovada
- **Contexto:** Bancos vetoriais otimizam busca por similaridade semântica, mas fragmentam documentos normativos em chunks. Em regulamentação aeronáutica, omitir um parágrafo de exceção ou alterar pontuação compromete a validade técnica e jurídica da orientação.
- **Decisão:** Utilizar o **ChromaDB** exclusivamente como índice de recuperação semântica e filtragem por metadados, mantendo o texto normativo oficial validado em arquivos JSON estruturados no volume `regulatory_documents` (*Single Source of Truth* — SSOT).
- **Consequências:**
  - Transcrições oficiais (`TEXTO OFICIAL`) são lidas diretamente do SSOT em Python, sem jamais passar por geração ou reescrita do LLM.
  - Exige manutenção de consistência entre o JSON canônico e os índices do ChromaDB (resolvida via pipeline idempotente com `SHA-256`).

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

## ADR-07 — SQLite (SSOT + FTS5) + ChromaDB 1.5.9 com Fusão RRF (Etapas 2 e 3)

- **Status:** Aprovada
- **Contexto:** O armazenamento canônico evoluiu para SQLite 3 (modo WAL, `FOREIGN KEYs ON`, transações ACID) no volume `regulatory_documents`. Na Etapa 3, consultas regulatórias precisam localizar termos exatos com e sem acentuação em português (ex.: `manutencao` vs. `manutenção`), referências estruturais (`RBAC 43.3`, `Art. 1º`) e perguntas semânticas abertas.
- **Decisão:**
  - Adotar índice lexical **SQLite FTS5** com tokenizador `unicode61 remove_diacritics 2` na migração incremental `v002_fts5_and_chunks.sql`, preservando `original_text` intacto no SSOT.
  - Combinar busca estrutural exata (`section_resolver`), busca lexical FTS5 (BM25) e busca vetorial no ChromaDB 1.5.9 via **Reciprocal Rank Fusion (RRF)**.
  - Hidratar todas as evidências ao vivo a partir do SQLite aplicando `ValidationService` (`review_on_demand`), impedindo que páginas OCR pendentes ou reprovadas fundamentem respostas.
- **Consequências:**
  - Nenhuma dependência externa adicional (FTS5 é nativo no Python 3.12 / SQLite 3.53).
  - Aprovações ou reprovações de páginas refletem imediatamente nas consultas sem necessidade de reindexar vetores.

---

## ADR-08 — SDK Oficial `google-genai`, Modelos Ativos e Guarda de Compatibilidade de Embeddings

- **Status:** Aprovada
- **Contexto:** O legado `google-generativeai` foi descontinuado em favor de `google-genai`; adicionalmente, o modelo `text-embedding-004` foi encerrado em janeiro/2026 e `gemini-2.5-flash` encontra-se em processo de descontinuação com roteamento para `gemini-3.5-flash-lite`.
- **Decisão:**
  - Adotar exclusivamente o SDK oficial `google-genai` (`v3.x`), mantendo `GEMINI_GENERATION_MODEL` (padrão `gemini-3.5-flash-lite`, com alias opcional `GEMINI_CHAT_MODEL`), `GEMINI_EMBEDDING_MODEL` (padrão `gemini-embedding-001`) e `GEMINI_EMBEDDING_DIMENSIONS` (padrão `768`, MRL) 100% configuráveis por ambiente.
  - Registrar modelo e dimensionalidade nos metadados da coleção ChromaDB e em `vector_index_registry` (SQLite), bloqueando mistura incompatível (`EmbeddingCompatibilityError`) e oferecendo reconstrução controlada (`rebuild_vector_index`) a partir do SQLite SSOT.
- **Consequências:**
  - Troca futura de modelo de geração ou embeddings ocorre sem reescrita de código e sem risco silencioso de degradação vetorial.

