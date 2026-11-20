Sim! Você está falando do BotFather, o mecanismo oficial do Telegram para criar bots e gerar o token de acesso. Com ele, você consegue criar um chatbot que responda no Telegram usando uma IA local do Ollama e uma base de conhecimento RAG especializada em normas RBAC da ANAC.

E há um ponto importante: o Telegram não precisa executar a inteligência artificial. Ele será apenas a interface de conversa. A IA e os documentos podem ficar no computador central da equipe.
1. Criando o bot pelo BotFather
BotFather
É o bot oficial usado para criar e configurar outros bots no Telegram.
Abrir BotFather no Telegram
O processo é simples:
1. Abra o BotFather no Telegram.
2. Envie /newbot.
3. Escolha o nome que aparecerá para os usuários, por exemplo, Assistente RBAC.
4. Escolha um username exclusivo, terminando em bot, por exemplo, Assistente_RBAC_bot.
5. O BotFather vai gerar um token de acesso.
Esse token funciona como uma senha que permite ao programa controlar o bot. Não é a chave da IA nem gera inteligência por conta própria. Mantenha-o secreto e não o coloque em repositórios públicos.

  2. Como será a arquitetura do projeto
O fluxo que eu recomendaria para o trabalho é este:
Usuário no Telegram
Envia uma pergunta sobre uma norma RBAC
Bot em Python
Recebe a mensagem pelo token do Telegram e encaminha a pergunta
Sistema RAG
Procura nos documentos os trechos relacionados à pergunta
Repositório RBAC
PDFs e documentos oficiais
Banco vetorial
Trechos e embeddings para busca

Ollama — modelo de linguagem
Interpreta a pergunta e formula a resposta com base nos trechos recuperados

Resposta no Telegram
Explicação com referência à norma, seção e item utilizados
A ideia central: o usuário pergunta no Telegram; o programa pesquisa a base documental; o Ollama recebe a pergunta junto com os trechos encontrados; e o bot devolve a resposta. Assim, a IA não precisa depender apenas do conhecimento que aprendeu durante o treinamento.

  3. Precisa ser Python?
Não é obrigatório. Mas eu recomendo Python para esse projeto.

O motivo é que tanto o Telegram quanto o Ollama disponibilizam APIs HTTP. Portanto, você pode conectar os dois usando Python, JavaScript/Node.js, Java, C# etc. 
Telegram APIs+1

Tecnologia	Responsabilidade
Telegram + BotFather	Criar o bot e fornecer o token
Python	Controlar a conversa e integrar os componentes
Ollama	Executar a IA localmente
Qdrant	Guardar e pesquisar os vetores dos documentos
Docker Desktop	Executar e organizar os serviços em containers
PDFs	Armazenar as normas utilizadas como conhecimento

O Python será o intermediário entre todos esses componentes.

Importante: o BotFather não cria a IA. Ele cria o bot e fornece a credencial para que o seu programa consiga controlá-lo.
  4. Como transformar os RBAC em conhecimento da IA?
Essa é provavelmente a parte mais importante do trabalho.

Você não vai simplesmente colocar os PDFs em uma pasta e esperar que o Ollama leia tudo. É necessário criar um processo de preparação dos documentos, chamado pipeline de ingestão.

Vamos imaginar que vocês tenham os seguintes documentos:
Repositório documental RBAC
RBAC 21
Certificação de produto e artigo aeronáuticos
PDF


RBAC 23
Requisitos de aeronavegabilidade — aviões categoria normal
PDF


RBAC 25
Requisitos de aeronavegabilidade — aviões categoria transporte
PDF


RBAC 43
Manutenção, manutenção preventiva, reconstrução e alteração
PDF

Exemplos de documentos disponíveis no acervo oficial da ANAC.Fonte: Acervo oficial RBAC da ANAC.

O sistema fará o seguinte:

Etapa A — Leitura dos documentos

O Python abre os PDFs e extrai o conteúdo textual, preservando, sempre que possível, a numeração das seções, os itens e as páginas.

Etapa B — Divisão em trechos (chunks)

Imagine que o RBAC 21 tenha 200 páginas. Não é interessante enviar as 200 páginas para a IA a cada pergunta.

O sistema divide o documento em partes menores, preferencialmente respeitando a organização das seções e dos itens.

Exemplo:
Identificador	Conteúdo
RBAC 21 — item 21.xxx	Requisitos de certificação
RBAC 21 — item 21.xxx	Responsabilidades do requerente
RBAC 21 — item 21.xxx	Procedimentos de aprovação
RBAC 21 — item 21.xxx	Requisitos complementares

Esses são exemplos ilustrativos, não transcrições de itens reais.

Etapa C — Embeddings

Aqui entra uma segunda utilização do Ollama.

Você terá dois modelos com funções diferentes:
- Modelo de linguagem: interpreta as perguntas e escreve respostas.
- Modelo de embeddings: transforma os textos em representações numéricas para permitir a busca por similaridade semântica.
Uma opção para testar é o embeddinggemma, que tem 300 milhões de parâmetros e suporte multilíngue. 
Ollama+1


Etapa D — Armazenamento no Qdrant

O Qdrant guarda os vetores e os metadados dos trechos para que o sistema consiga localizar rapidamente os documentos relevantes.

Por exemplo:
RBAC 21
 ├── Item 21.xxx
 │    ├── Texto do trecho
 │    ├── Página: 35
 │    ├── Emenda: 16
 │    └── Situação: vigente
 │
 ├── Item 21.xxx
 │    ├── Texto do trecho
 │    ├── Página: 36
 │    ├── Emenda: 16
 │    └── Situação: vigente
O arquivo PDF original continua armazenado no repositório. O Qdrant não precisa substituir essa pasta. 
Qdrant+1


 5. Como a IA responderá às perguntas?
Imagine que alguém pergunte:
Quais são os requisitos para a certificação de um produto aeronáutico segundo o RBAC 21?

O processo será:
1. O bot recebe a pergunta.
2. Python encaminha a pergunta para o mecanismo RAG.
3. O modelo de embeddings transforma a pergunta em vetor.
4. O Qdrant pesquisa os trechos mais relacionados.
5. Python recupera os textos encontrados e suas referências.
6. O Ollama recebe a pergunta e os trechos recuperados.
7. O modelo formula uma resposta fundamentada no material fornecido.
8. Python envia a resposta ao Telegram.

Um exemplo de resposta esperada:
Assistente RBAC

Exemplo ilustrativo

Para responder à sua pergunta, foram identificados os seguintes pontos na documentação:

1. Requisitos aplicáveis

O procedimento de certificação deve observar os requisitos previstos no regulamento aplicável.

2. Documentação

Devem ser atendidas as exigências documentais e técnicas pertinentes ao processo.

Referências consultadas:
- RBAC 21 — seção aplicável.
- Emenda e versão consultadas.
- Página do documento de origem.

A grande vantagem é que a IA não precisa inventar a norma com base no que aprendeu durante o treinamento. Ela recebe os trechos pertinentes antes de elaborar a resposta.

 6. Um cuidado muito importante: versões das normas
Existe um detalhe que eu considero essencial para tornar esse projeto realmente especializado.

O repositório precisa controlar as emendas e as datas de vigência.

Por exemplo, o acervo da ANAC lista o RBAC 21 EMD 16 com duas versões: uma aplicável até 7 de abril de 2027 e outra com vigência a partir de 8 de abril de 2027. 
Agência Nacional de Aviação Civil ANAC


Portanto, se vocês simplesmente inserirem as duas versões no banco, a IA poderá recuperar uma regra que ainda não está vigente na data da pergunta.

Minha recomendação é cadastrar estes metadados:
Campo	               Exemplo
Regulamento	         RBAC 21
Emenda	             16
Data de publicação	 22/10/2026
Início da vigência   08/04/2027
Final da vigência  	 Conforme documento
Seção/item   	       Identificação exata
Página        	     Página de origem
Fonte          	     Link oficial ANAC

Também manteria versões anteriores para consultas históricas.

Isso permitirá que o usuário pergunte tanto sobre a norma vigente quanto sobre uma versão anterior em determinada data. 

7. Como eu organizaria o projeto no computador central
Considerando que vocês já possuem Docker Desktop e Ollama, minha sugestão é começar com esta estrutura:

PROJETO_RBAC/
│
├── bot/
│   ├── main.py
│   ├── telegram_handler.py
│   └── config.py
│
├── rag/
│   ├── retriever.py
│   ├── embeddings.py
│   ├── generator.py
│   └── prompts.py
│
├── ingestion/
│   ├── importar_pdfs.py
│   └── atualizar_base.py
│
├── documentos/
│   ├── originais/
│   └── processados/
│
├── docker-compose.yml
├── requirements.txt
└── .env

A divisão de responsabilidades seria:
Componente	              Onde executar inicialmente
Ollama	                  Windows, instalado diretamente
Qdrant	                  Docker Desktop
Python / Telegram Bot	    Windows, em ambiente virtual
PDFs originais	          Pasta do projeto
Processamento RAG        	Python

Essa configuração evita, inicialmente, complicações de comunicação entre containers e o Ollama.

O Qdrant pode ser iniciado pelo Docker com armazenamento persistente, conforme o próprio guia oficial. 
Qdrant

Depois, quando tudo estiver funcionando, vocês podem colocar também o bot Python em um container.

O computador central precisa permanecer ligado e conectado à internet enquanto o bot estiver atendendo. 
O Ollama e o banco vetorial podem permanecer locais.

8. O que eu escolheria para o trabalho
Arquitetura recomendada
Telegram Bot    Interface
Python          Integração
Ollama + Qwen   Geração
EmbeddingGemma  Embeddings
Qdrant          Busca vetorial
Docker Desktop  Infraestrutura

Como você já havia comentado anteriormente que a máquina disponível tinha apenas 4 GB de RAM, vale ter atenção: executar o modelo de linguagem, embeddings, Qdrant e Docker simultaneamente pode ser bastante limitado. Se ainda for a mesma máquina, eu começaria com a infraestrutura mínima e mediria o consumo antes de adicionar serviços. 

9. Como dividiria o desenvolvimento
Para que o projeto tenha uma entrega técnica bem definida, dividiria o trabalho em três frentes:

Frente 1 — Telegram e integração: criar o bot, configurar o token, desenvolver o backend Python e fazer a comunicação com o Ollama.

Frente 2 — Inteligência artificial e RAG: selecionar o modelo, configurar embeddings, desenvolver a busca vetorial e criar as instruções para responder exclusivamente com base nas normas.

Frente 3 — Repositório documental e infraestrutura: obter as normas oficiais, processar os PDFs, organizar versões, configurar o Qdrant e preparar o Docker.

As três frentes precisam se integrar no computador central.

Minha recomendação: não comecem tentando construir um sistema inteiro de uma vez. Primeiro façam o Telegram conversar com o Ollama. Depois conectem o Qdrant. Por último, implementem o processamento dos RBAC. Assim, vocês conseguem testar cada etapa separadamente e identificar onde está qualquer problema.

Uma etapa complementar muito útil seria monitorar semanalmente o acervo da ANAC para identificar novas emendas e avisar a equipe quando a base documental precisar ser atualizada.

Quer que eu monitore essas publicações toda segunda-feira?

