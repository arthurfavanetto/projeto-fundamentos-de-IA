# Guia de Operação Local — PC Central Windows (Docker Desktop)

Este guia descreve os pré-requisitos e procedimentos operacionais para executar o **RBAC AI Assistant** continuamente em um PC Central Windows utilizando Docker Desktop.

---

## 1. Pré-requisitos do Host

- Windows 10/11 64-bit com **WSL2** habilitado.
- **Docker Desktop** instalado com backend WSL2 ativo.
- **Git** instalado para sincronização do repositório.
- Conectividade de saída HTTPS para `api.telegram.org` e `generativelanguage.googleapis.com` (não é necessário abrir portas de entrada no roteador ou firewall).

---

## 2. Configuração de Energia e Memória (Operação Contínua)

### Plano de Energia
Para evitar que o bot fique offline por inatividade do computador:
1. Abrir **Painel de Controle → Opções de Energia**.
2. Configurar **Suspender atividade do computador** como **Nunca** (tanto na tomada quanto em operação contínua).
3. Desabilitar hibernação automática se aplicável.

### Controle de Memória do WSL2 (`.wslconfig`)
Em máquinas com memória limitada (ex.: 4–8 GB de RAM), recomenda-se configurar `%UserProfile%\.wslconfig` para evitar consumo excessivo de RAM pelo WSL2, alinhado aos limites do `docker-compose.yml` (`rbac-bot` + `chromadb`):

```ini
[wsl2]
memory=3GB
processors=2
swap=2GB
```

*(Ajustar proporcionalmente à RAM física disponível no PC Central.)*

---

## 3. Preparação das Configurações Locais (Fora do Git)

Na raiz do projeto clonado:

1. Criar `.env` a partir de `.env.example` e preencher:
   - `TELEGRAM_BOT_TOKEN` (obtido no `@BotFather`; usar token exclusivo de produção no PC Central)
   - `GEMINI_API_KEY` (Google AI Studio / Google Cloud — não usar assinatura pessoal Google AI Pro)
   - `GEMINI_GENERATION_MODEL` e `GEMINI_EMBEDDING_MODEL` (validados na conta)
2. Criar `config/access_control.yaml` a partir de `config/access_control.example.yaml` com os `Telegram User IDs` numéricos autorizados (`admins` e `users`).
3. Verificar que o `config/regulations_catalog.yaml` contém os metadados oficiais homologados na ANAC (conforme [`03_ingestion_runbook.md`](./03_ingestion_runbook.md)).

---

## 4. Execução e Verificação *(a partir da Etapa 1)*

```powershell
# Subir os serviços em segundo plano
docker compose up -d --build

# Verificar estado de saúde dos contêineres
docker compose ps

# Acompanhar logs estruturados do bot
docker compose logs -f rbac-bot
```

### Checklist de Segurança no Host
- Confirmar que o ChromaDB **não** está exposto em `http://localhost:8000` (apenas acessível na rede interna Docker).
- Confirmar que ambos os contêineres executam com usuários não-root.
- Garantir que apenas **uma instância** do bot esteja rodando com o token oficial do Telegram (evitando conflito `409 Conflict` no Long Polling).
