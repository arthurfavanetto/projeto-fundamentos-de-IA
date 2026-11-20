Com 4 pessoas, eu dividiria o projeto em 4 frentes de responsabilidade, cada uma com um integrante responsável por uma parte específica do sistema. Assim, todos conseguem trabalhar em paralelo, mas o resultado final será um único assistente RBAC integrado ao Telegram.

Minha sugestão é não dividir apenas por ferramentas, mas por funções do sistema.
1. Divisão dos 4 integrantes
01

Integrante 1 — Bot Telegram e Backend
Comunicação


Responsabilidade: desenvolver a comunicação entre o usuário, o Telegram e o restante do sistema.

Atividades:
- Criar o bot pelo BotFather e configurar o token.
- Desenvolver o código Python do chatbot.
- Receber e interpretar as mensagens dos usuários.
- Encaminhar as perguntas para o sistema RAG.
- Receber as respostas e devolvê-las ao Telegram.
- Implementar comandos como /start, /help e tratamento de erros.
- Controlar o histórico de conversas, se necessário.
Entrega: um bot funcional no Telegram que consegue receber perguntas e apresentar respostas de um serviço de teste.


  02

Integrante 2 — Inteligência Artificial e Ollama
Modelo de IA


Responsabilidade: desenvolver a inteligência responsável por interpretar as perguntas e elaborar respostas.

Atividades:
- Configurar o Ollama.
- Escolher e testar o modelo de linguagem, por exemplo, Qwen.
- Configurar o modelo de embeddings em conjunto com o responsável pelo RAG.
- Desenvolver o prompt de sistema para o especialista RBAC.
- Configurar regras para impedir respostas sem fundamentação documental.
- Fazer a IA interpretar os trechos recuperados.
- Configurar respostas com referências aos itens e páginas das normas.
Entrega: um modelo funcionando localmente, capaz de responder perguntas utilizando textos fornecidos pelo sistema e respeitando as instruções do especialista RBAC.


03

Integrante 3 — Engenharia RAG e Banco Vetorial
Busca inteligente


Responsabilidade: desenvolver o mecanismo que permite à IA pesquisar e recuperar informações relevantes nos documentos.

Atividades:
- Desenvolver o processo de divisão dos documentos em chunks.
- Configurar os embeddings.
- Configurar o Qdrant.
- Armazenar os vetores e metadados.
- Desenvolver a busca semântica.
- Recuperar os trechos mais relevantes para cada pergunta.
- Entregar os trechos encontrados ao modelo de linguagem.
- Testar a qualidade das pesquisas.
Entrega: mecanismo RAG funcionando, capaz de receber uma pergunta e devolver os trechos relevantes das normas, incluindo suas referências.

  
04

Integrante 4 — Repositório Documental, Infraestrutura e Validação
Documentação e qualidade


Responsabilidade: preparar o conhecimento utilizado pelo especialista e organizar o ambiente de execução e os testes.

Atividades:
- Pesquisar e obter os RBAC em fontes oficiais da ANAC.
- Organizar os PDFs no repositório documental.
- Identificar emendas, datas de publicação e vigência.
- Verificar a qualidade dos documentos e a extração dos textos.
- Preparar os metadados e disponibilizar os documentos para ingestão no RAG.
- Configurar o Docker Desktop e o ambiente dos serviços.
- Preparar os testes de precisão das respostas e das referências.
- Documentar a instalação e o funcionamento do projeto.
Entrega: repositório documental organizado, infraestrutura executável e relatório de testes demonstrando a qualidade do especialista.



2. Como as quatro partes se conectam
O ponto mais importante é entender que ninguém trabalha completamente isolado. Cada integrante entrega uma parte que será utilizada pelo próximo.
Fluxo de integração da equipe
Integrante 4 — Documentação
Prepara os RBAC oficiais, PDFs e metadados

Integrante 3 — RAG
Processa documentos, cria embeddings e realiza pesquisas no Qdrant

Integrante 2 — IA
Recebe pergunta e trechos encontrados, interpreta e gera resposta fundamentada

Integrante 1 — Backend e Telegram
Gerencia a conversa e envia a resposta ao usuário

Usuário recebe a resposta no Telegram


Na prática, os integrantes 2 e 3 precisam trabalhar bastante próximos: o modelo de embeddings e o modelo de linguagem são coisas diferentes, mas precisam funcionar juntos.


3. Como dividir o trabalho ao longo do desenvolvimento
Eu faria a execução em quatro etapas, para evitar que alguém fique parado esperando outra pessoa.
Etapa	                     Integrante 1	                          Integrante 2	                         Integrante 3	                Integrante 4
1. Preparação	             Criar bot	                            Configurar Ollama                      Estudar RAG/Qdrant	          Buscar RBAC e configurar Docker
2. Desenvolvimento	       Backend Python	                        Prompt e modelo	                       Embeddings e busca	          Extração e organização dos PDFs
3. Integração	             Conectar ao RAG	                      Integrar geração	                     Conectar documentos e IA	    Preparar ambiente e validar fontes
4. Testes	                 Testar Telegram	                      Avaliar respostas	                     Avaliar recuperação	        Validar referências e documentar

Uma observação: o integrante 4 não deve esperar o RAG ficar pronto para organizar os documentos. Ele pode começar a montar o repositório e testar a extração dos PDFs desde o primeiro dia.  



4. Como evitar problemas na hora de juntar os códigos
Essa é uma parte que costuma causar problemas em projetos com várias pessoas.

Eu estabeleceria desde o início um padrão de comunicação entre os componentes.

Por exemplo, o integrante 1 não precisa saber como o Qdrant funciona. Ele só precisa saber como solicitar uma resposta ao sistema.

O backend poderia fazer uma chamada como:
resultado = especialista.perguntar(
    pergunta="Quais são os requisitos do RBAC 21?"
)
E receber algo como:
{
    "answer": "Resposta fundamentada na norma...",
    "sources": [
        {
            "regulation": "RBAC 21",
            "section": "Item aplicável",
            "page": 35,
            "amendment": "16"
        }
    ]
}

Isso permite que cada integrante desenvolva sua parte sem precisar alterar constantemente o código dos outros.

Também recomendo utilizar um repositório GitHub compartilhado, com pastas e responsabilidades definidas.

5. O que cada integrante precisa aprender
Integrante	Conhecimentos principais
1	Python, Telegram Bot API e requisições HTTP
2	Ollama, modelos LLM, engenharia de prompts
3	RAG, embeddings, chunking e Qdrant
4	PDFs, Python, metadados, Docker e validação documental 

Não é necessário que todos dominem todas as tecnologias. Entretanto, todos devem compreender o fluxo geral do projeto para conseguir explicar o sistema na apresentação.
  
  6. Minha recomendação para a equipe
Eu escolheria um responsável geral pela integração técnica, preferencialmente o integrante 1, porque ele será o responsável por conectar o chatbot às outras partes.

Também estabeleceria três requisitos obrigatórios para considerar o projeto concluído:
1. Fundamentação: toda resposta sobre RBAC deve apresentar as referências documentais utilizadas.
2. Confiabilidade: quando não encontrar informação suficiente nos documentos, a IA deve admitir que não encontrou fundamentação, em vez de inventar uma resposta.
3. Rastreabilidade: o sistema deve identificar regulamento, emenda, item, página e vigência da fonte utilizada.



  
