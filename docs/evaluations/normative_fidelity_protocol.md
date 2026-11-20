# Protocolo de Avaliação — Fidelidade Normativa e Grounding de IA

Este documento define os critérios objetivos de validação para garantir que o **RBAC AI Assistant** reproduza fielmente o texto oficial da ANAC e produza explicações tecnicamente fundamentadas sem alucinações.

---

## 1. Dataset Áureo (*Golden Dataset*)

O dataset áureo residirá localmente em `data/golden_dataset/` (fora do Git quando contiver trechos integrais dos PDFs oficiais, ou com fixtures sintéticas em `tests/` para CI) e será construído a partir de seções reais do **RBAC 43** conferidas manualmente caractere por caractere contra o PDF oficial da ANAC.

Cada amostra do dataset áureo contém:
- `regulation_number` e `amendment_number`
- `section_number` (incluindo seções curtas, longas, com parágrafos/incisos/alíneas e apêndices)
- `exact_text`: transcrição oficial validada manualmente
- `pdf_page_number` e `printed_page_number`
- `expected_key_obligations`: lista de obrigações/exceções que devem constar na explicação

---

## 2. Critérios de Fidelidade Literal (`tests/evals/test_normative_fidelity.py`)

- **Asserção estrita:** `response.official_text == golden_sample.exact_text`.
- **Tolerância zero** para:
  - Alteração de verbos normativos (*deve*, *pode*, *exceto*, *somente*)
  - Perda ou alteração de numeração (`(a)`, `(1)`, `(i)`, `(A)`)
  - Mutilação de pontuação (`;`, `:`, `.`, `—`)
  - Divergência entre `pdf_page_number` e `printed_page_number`
- **Validação de paginação Telegram:** concatenar todas as mensagens produzidas pelo `MessageSplitter` (removendo apenas cabeçalhos de paginação `[Parte X/Y]`) deve reconstituir **100% exatos** da transcrição original.
- Uma seção só recebe o selo `extraction_validation_status = "validated"` após aprovação neste protocolo.

---

## 3. Cenários Críticos de Qualidade e Grounding (`tests/evals/test_ai_grounding_quality.py`)

| # | Cenário | Comportamento Esperado |
| :-: | :--- | :--- |
| 1 | **Pergunta Direta por Seção** (ex.: `/rbac 43.3`) | Recupera texto literal exato do SSOT; explicação IA sem contradições com o dispositivo. |
| 2 | **Pergunta Aberta Conceitual** (ex.: retorno ao serviço) | Todos os requisitos citados na explicação existem efetivamente nas seções recuperadas. |
| 3 | **Cruzamento de Requisitos** (ex.: execução vs. registro de manutenção) | Recupera e cita explicitamente ambas as seções aplicáveis com suas páginas. |
| 4 | **Ausência de Fundamentação (Teste Negativo)** (ex.: seção `43.999` ou tema de outro RBAC não indexado) | Declara explicitamente ausência de evidência na base indexada; **zero invenção de artigos ou requisitos externos (ex.: FAA 14 CFR Part 43)**. |
| 5 | **Comparação entre Emendas** | Quando múltiplas emendas existem, separa explicitamente `[EMENDA VIGENTE]` de `[EMENDA ANTERIOR]` sem fundir redações. |
