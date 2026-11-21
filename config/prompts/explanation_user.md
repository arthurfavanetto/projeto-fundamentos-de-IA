# User Prompt Template — Explicação Normativa Fundamentada v1.0

Modo de fundamentação: {grounding_mode}
Aviso de segurança: {security_notice}

<user_question>
{user_question}
</user_question>

<untrusted_normative_evidence>
{evidence_blocks}
</untrusted_normative_evidence>

## Instruções para a Explicação

- Produza apenas a seção de **Explicação** em português do Brasil, clara e direta.
- Fundamente cada afirmação estritamente nas evidências dentro de `<untrusted_normative_evidence>`.
- Não repita blocos longos do texto literal oficial (ele já é entregue separadamente pelo sistema determinístico) e nunca apresente paráfrase como citação literal.
- Se `grounding_mode` for `preliminary_unreviewed`, deixe claro no início que se trata de localização preliminar baseada em extração ainda não validada manualmente.
- Se as evidências não forem suficientes para responder à pergunta, informe isso de forma objetiva sem extrapolar.
