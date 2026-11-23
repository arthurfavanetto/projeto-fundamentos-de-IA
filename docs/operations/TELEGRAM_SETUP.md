# Guia Operacional — Bot Telegram (Etapa 4) e Controle de Acesso

Este documento detalha a arquitetura implementada, a configuração segura de credenciais e da lista de acesso (`deny_by_default`) e o roteiro de validação local do **Bot Telegram** no Windows 11.

> **Estado de validação:** todos os handlers, controle de acesso `deny_by_default`, `UserRateLimiter`, `MessageSplitter` (4.096 UTF-16), tratamento de erros da API Telegram e encerramento gracioso estão implementados e testados **100% offline com mocks** (VM Linux e GitHub Actions CI). O teste real com o servidor da API Telegram e usuários autorizados permanece **PENDENTE DE VALIDAÇÃO LOCAL NO WINDOWS**.

---

## 1. Arquitetura e Segurança Implementadas

- **Dependência:** `python-telegram-bot` (v22.x, fixada em `pyproject.toml` e `uv.lock`, modo **HTTPS Long Polling** — sem webhooks nem portas de entrada no roteador, ADR-03).
- **Controle de acesso determinístico (`src/rbac_assistant/security/access_control.py`):**
  - Política imutável **`deny_by_default`**: qualquer `Telegram User ID` ausente é bloqueado antes de qualquer chamada ao RAG ou Gemini.
  - Se `config/access_control.yaml` estiver ausente, malformado ou vazio, **100% dos usuários são bloqueados** (modo *fail-closed*, sem usuários ou administradores fictícios).
  - Aplicado uniformemente a todos os 8 comandos, comandos não reconhecidos e mensagens de texto livre.
- **Proteções operacionais:**
  - `UserRateLimiter`: `15` requisições/minuto e rajada (`burst_size`) de `5` por usuário.
  - Limite de entrada: `1000` caracteres por mensagem (`telegram.max_input_length`).
  - Deduplicação de `update_id` (cache LRU de `512` eventos) e controle de concorrência por usuário.
  - `MessageSplitter`: paginação fiel `<= 4.096` unidades UTF-16 com cabeçalhos `[Parte X/Y]` preservando 100% do texto normativo literal e aplicando redação de segredos (`secret_redactor`).
  - Modo standby seguro: quando `TELEGRAM_BOT_TOKEN` está ausente/placeholder (ex.: CI ou testes offline) ou `TELEGRAM_POLLING_ENABLED=false`, o contêiner `rbac-bot` permanece saudável para CLIs e healthchecks sem tentar conexão externa.

---

## 2. Comandos Suportados

| Comando | Perfil mínimo | Comportamento |
| --- | :---: | --- |
| `/start` | `USER` | Boas-vindas, perfil autorizado e explicação da separação literal vs. IA |
| `/help` | `USER` | Lista completa de comandos suportados |
| `/status` | `USER` / `ADMIN` | Saúde do ChromaDB e índices (`ADMIN` exibe também heartbeat, volumes graváveis e coleção) |
| `/rbac <seção>` | `USER` | Recuperação direta do SQLite SSOT sem geração IA (ex.: `/rbac 43.3`) |
| `/buscar <termo>` | `USER` | Busca híbrida (estrutural + FTS5 + vetorial) sem geração IA |
| `/explicar <seção ou pergunta>` | `USER` | Fluxo completo RAG: texto normativo literal aprovado + explicação Gemini + referências |
| `/documentos` | `USER` | Lista regulamentos cadastrados e status de indexação |
| `/versao` | `USER` | Versão do sistema, política Gemini Free Tier e emendas cadastradas |
| Texto livre | `USER` | Equivalente a consulta completa via `NormativeChatService` |

---

## 3. Configuração Local Segura (Host Windows 11)

### 3.1. Configurar `config/access_control.yaml` (fora do Git)

O arquivo `config/access_control.yaml` é montado **somente para leitura** (`./config:/app/config:ro`) no contêiner `rbac-bot` e permanece bloqueado no `.gitignore`.

```powershell
Set-Location "C:\Projetos\projeto-fundamentos-de-IA"
Copy-Item .\config\access_control.example.yaml .\config\access_control.yaml
git check-ignore -v config/access_control.yaml
```

Edite `config\access_control.yaml` substituindo os exemplos pelos **Telegram User IDs numéricos reais** dos operadores autorizados (remova IDs não utilizados):

```yaml
policy:
  deny_by_default: true

roles:
  admins:
    - <SEU_TELEGRAM_USER_ID_NUMERICO>
  users: []
```

### 3.2. Conferir `TELEGRAM_BOT_TOKEN` no `.env` local (sem exibir o valor)

```powershell
# O .env local já deve conter TELEGRAM_BOT_TOKEN=<token do @BotFather>
# Nunca cole ou imprima o valor do token no terminal ou no chat.
docker compose up -d --build
docker compose ps
docker compose logs --tail=40 rbac-bot
```

---

## 4. Roteiro de Teste Real Controlado (No Telegram do Proprietário)

1. Confirme nos logs de `rbac-bot` o evento `telegram_long_polling_started` (sem `telegram_invalid_bot_token` nem `409 Conflict`).
2. Com um usuário **autorizado** em `config/access_control.yaml`, teste em sequência:
   - `/start` → confirmar perfil `ADMIN` ou `USER`.
   - `/status` → confirmar `SAUDÁVEL` e contagens corretas.
   - `/documentos` e `/versao` → confirmar `ANAC RBAC 43` (após ingestão) e política `gemini-4-hydrogen` / `gemini-embedding-2 (768d)`.
   - `/rbac 43.3` e `/buscar manutencao` → confirmar transcrição/localização sem erro.
   - `/explicar 43.3` e uma pergunta em texto livre → confirmar blocos separados e citações válidas.
3. Com um usuário **NÃO listado** em `config/access_control.yaml` (ou removendo temporariamente um ID de teste), envie `/start`, `/rbac 43.3` e texto livre → confirmar resposta `Acesso negado` e registro `telegram_access_control` (`outcome: "denied"`) em `/app/logs/audit_events.jsonl`.
