Essa é uma das decisões mais importantes da organização do projeto. Minha recomendação é: todos os quatro integrantes podem desenvolver simultaneamente, mas o bot oficial deve funcionar em uma única instância no computador central.

Não é necessário revezar o uso do Telegram nem o token. Vou explicar a diferença.
1. Primeiro: existem três situações diferentes
Situação	Pode acontecer simultaneamente?
Quatro integrantes programando o projeto	Sim
Vários usuários perguntando ao bot no Telegram	Sim
Quatro computadores executando o mesmo bot com o mesmo token para receber mensagens	Não recomendado

O segredo está em diferenciar o desenvolvimento do programa de sua execução.
2. Como os quatro integrantes trabalham ao mesmo tempo?
Imaginem que cada integrante esteja em seu próprio computador.


Computadores da equipe
Pessoa 1
Bot Telegram
Pessoa 2
Ollama e IA
Pessoa 3
RAG e Qdrant
Pessoa 4
Documentos e Docker

GitHub — Repositório compartilhado
Todos enviam suas alterações para o mesmo projeto
Computador central
Versão integrada
Recebe os códigos, executa a IA, o RAG e o bot oficial

Para isso, recomendo utilizar o GitHub.

Cada pessoa terá uma cópia do projeto no próprio computador. Quando terminar uma funcionalidade, envia suas alterações para o repositório compartilhado.

Por exemplo:
- Pessoa 1 trabalha na pasta bot/.
- Pessoa 2 trabalha na pasta ia/.
- Pessoa 3 trabalha na pasta rag/.
- Pessoa 4 trabalha na pasta documentos/ e na configuração de infraestrutura.
Assim, os quatro podem programar no mesmo horário, sem precisar acessar simultaneamente o computador central.

3. Como juntar tudo no computador central?
O computador central será o servidor do projeto.

Ele terá uma única versão integrada, com todos os componentes instalados.

Eu organizaria assim:
Computador central — servidor RBAC
Bot Telegram   Python
Recebe mensagens e controla as conversas dos usuários.

Ollama    Local
Executa o modelo de linguagem e o modelo de embeddings.

Qdrant                      Docker
Armazena vetores e metadados para a pesquisa documental.

Repositório RBAC    Arquivos
Contém os PDFs oficiais, versões, emendas e documentos processados.

Token do Telegram
Armazenado somente no ambiente do servidor.


Uma configuração inicial simples seria:
Componente	Execução
Ollama	Instalado no Windows
Qdrant	Container no Docker Desktop
Python / Bot	Executado no Windows
Documentos	Pasta do computador central
Token	Arquivo de configuração protegido

O Docker consegue organizar os serviços e suas redes, permitindo que os containers se comuniquem.


4. Agora, sobre o token do Telegram
Aqui está a parte mais importante da sua pergunta.

O token identifica e autoriza o acesso ao bot. Ele funciona como uma credencial de controle.

Imagine que vocês criaram:

@AssistenteRBAC_bot

O BotFather fornece um token único.

Esse token será utilizado pelo programa Python que fica no computador central.

Não é necessário que os quatro integrantes tenham esse token.

Na verdade, recomendo que apenas o responsável pela integração e o administrador do projeto tenham acesso à credencial real.

O Telegram deixa claro que qualquer pessoa com o token pode controlar o bot. 
Telegram APIs+1

E se os quatro executarem o bot ao mesmo tempo?
Imagine esta situação:
- Computador 1 executa main.py.
- Computador 2 executa main.py.
- Computador 3 executa main.py.
- Computador 4 executa main.py.
Todos usando o mesmo token e o método de recebimento long polling.

Isso pode gerar conflitos na distribuição e confirmação das mensagens, porque todos estão tentando consumir as mesmas atualizações do bot.

O Telegram oferece dois métodos para receber atualizações: long polling e webhook. Não é possível utilizar long polling enquanto um webhook estiver configurado. 

Para o projeto de vocês, eu escolheria:

Recomendado  Uma única instância do bot rodando no computador central.

Assim, somente um processo recebe as atualizações do Telegram.

5. Mas várias pessoas podem conversar com o bot ao mesmo tempo?
Sim! E essa é uma distinção muito importante.

Vocês não precisam de quatro bots para atender quatro pessoas.

Um único bot pode atender:
- Integrante 1 fazendo uma pergunta sobre RBAC 21.
- Integrante 2 perguntando sobre RBAC 23.
- Integrante 3 perguntando sobre RBAC 25.
- Integrante 4 perguntando sobre RBAC 43.
Todas as mensagens chegam ao mesmo backend, que identifica cada conversa e encaminha a pergunta ao RAG.

O processamento pode ocorrer de maneira concorrente ou por uma fila de solicitações.
Exemplo de atendimento simultâneo
Usuário 1
RBAC 21
Usuário 2
RBAC 23
Usuário 3
RBAC 25
Usuário 4
RBAC 43


Uma única instância Python
Identifica cada usuário e gerencia suas solicitações

RAG + Qdrant + Ollama
Pesquisa os documentos e gera as respostas

Resposta 1 e 2

Resposta 3 e 4


Existe uma ressalva: o computador precisa ter capacidade para processar várias perguntas. Como você já havia mencionado uma máquina com 4 GB de RAM, eu começaria com uma fila para controlar as consultas ao Ollama, evitando que várias gerações pesadas comprometam o desempenho.

6. E como testar sem atrapalhar o bot oficial?
Uma solução muito boa é criar dois ambientes:
Ambiente	Finalidade	Token
Desenvolvimento	Testes individuais da equipe	Tokens de bots de teste
Produção / demonstração	Bot oficial RBAC	Token oficial

Por exemplo, a pessoa 1 cria um bot de testes pelo próprio BotFather. Ela consegue desenvolver a integração do Telegram sem interferir no bot que está sendo utilizado pela equipe.

O próprio Telegram recomenda utilizar outro bot para realizar testes sem interferir na instância original. 
Telegram APIs


Quando o código estiver pronto, ele é enviado ao GitHub e integrado à versão principal.

7. Organização que eu adotaria para vocês
Responsável	O que faz no computador próprio	O que entrega ao central
Pessoa 1	Programa o bot	Código Python e integração Telegram
Pessoa 2	Desenvolve a IA	Configuração do Ollama, modelos e prompts
Pessoa 3	Desenvolve o RAG	Busca vetorial e integração Qdrant
Pessoa 4	Organiza os RBAC e infraestrutura	Documentos, metadados e Docker

O responsável pela integração (Pessoa 1, por exemplo) controla a versão oficial, mas não precisa desenvolver tudo sozinho.

No computador central, a estrutura ficaria assim:

C:\PROJETO_RBAC\
│
├── bot\
│   └── main.py
│
├── ia\
│   └── generator.py
│
├── rag\
│   └── retriever.py
│
├── ingestion\
│   └── importar_pdfs.py
│
├── documentos\
│   ├── originais\
│   └── processados\
│
├── docker-compose.yml
├── requirements.txt
├── .env
└── .gitignore

O arquivo .env contém o token verdadeiro e outras configurações. Ele deve ficar fora do GitHub, protegido pelo .gitignore. No GitHub, vocês compartilham somente um .env.example sem credenciais reais.

Minha decisão para a equipe
Eu não faria rodízio de token nem de uso do bot.

Adotaria este modelo:
- Quatro pessoas: desenvolvendo simultaneamente em seus próprios computadores.
- Um GitHub: concentrando todos os códigos.
- Um computador central: executando o sistema integrado.
- Um token oficial: controlando o bot de produção.
- Uma instância Python: recebendo as mensagens do Telegram.
- Vários usuários: utilizando o bot simultaneamente.
Assim, vocês conseguem desenvolver em paralelo, testar separadamente e depois colocar tudo para funcionar em um único computador, sem que a equipe fique dependendo de uma pessoa liberar o Telegram para outra.
