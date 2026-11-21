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
- Fundamente cada afirmação estritamente nas evidências validadas dentro de `<untrusted_normative_evidence>` e referencie explicitamente os identificadores de evidência fornecidos (ex.: `[EV-1]`, `[EV-2]`).
- Nunca cite seções, parágrafos, emendas ou páginas que não estejam presentes nas evidências fornecidas; citações não confirmadas serão descartadas pelo validador determinístico do servidor.
- Não repita blocos longos do texto literal oficial (ele já é entregue separadamente pelo sistema determinístico a partir do SQLite) e nunca apresente paráfrase como citação literal.
- Se as evidências não forem suficientes para responder à pergunta, informe isso de forma objetiva sem extrapolar.
