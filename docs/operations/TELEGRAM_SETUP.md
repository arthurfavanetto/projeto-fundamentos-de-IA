# Guia Operacional — Bot Telegram Público, Dois Administradores e Escopo Normativo Estrito

Este documento detalha a arquitetura operacional do **Bot Telegram** em **modo público em conversas privadas**, a separação determinística de privilégios para **dois administradores**, o comando `/id`, as proteções contra abuso e o roteiro de publicação e validação local no Windows 11.

> **Estado de validação:** o modo público (`public_access: true`, `default_role: USER`), o isolamento dos dois administradores (`ADMIN`), o comando `/id`, o escopo estrito de conversas privadas (`allow_group_chats: false`), a recuperação restrita a páginas aprovadas (`only_approved_literal=True`), o `UserRateLimiter` (por usuário + rajada + global), a concorrência global, o timeout de chamadas, o `MessageSplitter` (4.096 UTF-16) e a compatibilidade `PTB_TIMEDELTA=1` estão implementados e testados **100% offline com mocks** (VM Linux e GitHub Actions CI). O teste real com a API do Telegram e a configuração local dos dois IDs em `config/access_control.yaml` permanecem **PENDENTES DE VALIDAÇÃO LOCAL NO WINDOWS**.

---

## 1. Política de Acesso Público e Separação de Privilégios

Por decisão expressa do proprietário, o bot opera em **modo público** para usuários comuns em conversas privadas, mantendo os privilégios administrativos restritos exclusivamente aos dois administradores cadastrados localmente:

- **Acesso público (`policy.public_access: true`, `policy.default_role: USER`):**
  - Qualquer pessoa que abrir uma conversa privada com o bot pode enviar comandos e perguntas sem cadastro, whitelist ou aprovação prévia, recebendo automaticamente o perfil **`USER`**.
  - Usuários comuns **nunca** recebem privilégios `ADMIN` — nem por ausência de `config/access_control.yaml`, nem por arquivo malformado/incompatível, nem por tentativa de definir `default_role: ADMIN`, nem por parâmetros/mensagens no chat.
- **Dois administradores (`roles.admins`):**
  - Recebem perfil **`ADMIN`** apenas os Telegram User IDs numéricos listados em `roles.admins` no arquivo local `config/access_control.yaml` (ignorado pelo Git).
  - Arquivo ausente concede apenas `USER` no modo público (`admins = ()`, zero administradores); arquivo malformado ou raiz inválida opera em modo *fail-closed* (`public_access=False`, zero administradores).
- **Desabilitação explícita do modo público:**
  - Caso o proprietário deseje fechar o acesso futuramente, basta definir `policy.public_access: false` em `config/access_control.yaml` (e opcionalmente informar `roles.users`).
- **Escopo inicial de conversas (`telegram.allow_group_chats: false`):**
  - A versão pública inicial atende **exclusivamente conversas privadas individuais** (`filters.ChatType.PRIVATE` + guarda em `TelegramBotRunner.handle_update`).
  - Mensagens gerais de grupos, supergrupos e canais são ignoradas automaticamente. O suporte a grupos permanece separado para decisão futura.

---

## 2. Comandos Suportados e Separação `USER` vs. `ADMIN`

| Comando | Perfil | Comportamento |
| --- | :---: | --- |
| `/start` | `USER` / `ADMIN` | Boas-vindas, perfil ativo, modo de acesso e explicação do escopo normativo estrito |
| `/help` | `USER` / `ADMIN` | Lista completa dos 9 comandos suportados e instruções de uso |
| `/id` | `USER` / `ADMIN` | Retorna **exclusivamente** o Telegram User ID numérico do próprio solicitante (nunca exibe IDs de terceiros nem registra o ID em logs públicos) |
| `/status` | `USER` (público) | Formato público sanitizado: disponibilidade geral, documentos/versões indexadas, disponibilidade da explicação IA e escopo normativo — **sem** caminhos internos, coleção vetorial, heartbeats, volumes ou credenciais |
| `/status` | `ADMIN` | Inclui `--- Diagnóstico ADMIN ---`: ambiente, política de acesso, quantidade de administradores configurados, status/heartbeat do ChromaDB, chunks SQLite/vetores Chroma, gravabilidade de volumes, coleção vetorial, versões não indexadas/defasadas e erros redigidos |
| `/rbac <seção>` | `USER` / `ADMIN` | Recuperação direta do texto literal aprovado no SQLite SSOT sem geração IA (ex.: `/rbac 43.3`) |
| `/buscar <termo>` | `USER` / `ADMIN` | Busca híbrida (estrutural + FTS5 + vetorial) restrita a páginas aprovadas, sem geração IA |
| `/explicar <seção ou pergunta>` | `USER` / `ADMIN` | Texto normativo literal aprovado + explicação Gemini + referências validadas |
| `/documentos` | `USER` / `ADMIN` | Lista regulamentos disponíveis (`ADMIN` vê contagem detalhada de chunks e dicas operacionais) |
| `/versao` | `USER` / `ADMIN` | Versão do assistente, política Gemini Free Tier e emendas vigentes (`ADMIN` vê também ambiente e coleção ChromaDB) |
| Texto livre | `USER` / `ADMIN` | Pergunta em linguagem natural respondida apenas com evidência aprovada e indexada (recusa fundamentada fora do escopo ou sem evidência) |

---

## 3. Escopo Normativo Estrito e Proteções Contra Abuso

- **Escopo normativo estrito (Telegram público):**
  - `telegram_only_approved_literal: true` restringe a recuperação estrutural, lexical (FTS5) e vetorial (ChromaDB) exclusivamente a páginas aprovadas (`VALIDATED_LITERAL`). Conteúdo aguardando revisão ou reprovado é excluído.
  - Sem busca na internet e sem conhecimento geral do Gemini para preencher lacunas.
  - `PromptGuard` trata tentativas de *prompt injection*, exfiltração de prompts/variáveis de ambiente/credenciais, escalada de privilégio e quebra de escopo como dados não confiáveis.
  - `CitationValidator` descarta deterministicamente `[EV-N]`, seções, artigos e páginas não confirmados nas evidências recuperadas; `secret_redactor` sanitiza qualquer padrão de segredo.
- **Proteções para operação pública:**
  - Entrada limitada a `1000` caracteres (`telegram.max_input_length`).
  - `UserRateLimiter`: `15` req/min por usuário, rajada (`burst_size`) de `5` em `10s` e teto global de `60` req/min (`security.rate_limit.global_requests_per_minute`) protegendo a cota Gemini Free Tier (`GEMINI_ALLOW_PAID_MODELS=false`, sem fallback pago).
  - Concorrência limitada a `1` atualização simultânea por usuário e `8` globais (`telegram.max_global_concurrent_updates`), com timeout de `45s` (`telegram.command_timeout_seconds`) e deduplicação LRU de `512` `update_id`s.
  - Isolamento completo entre usuários: **sem memória conversacional cruzada** e sem exposição de auditoria de terceiros.
  - `security.audit.pseudonymize_user_ids: true` pseudonimiza `actor_id` no JSONL privado em `/app/logs/audit_events.jsonl`, e logs `stdout/stderr` não registram IDs de usuários indiscriminadamente.

---

## 4. Como Obter os IDs dos Dois Administradores e Configurar Localmente (Windows 11)

> **Segurança:** nunca envie `TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY` ou Telegram User IDs pessoais no chat nem faça commit de `config/access_control.yaml` ou `.env`.

### 4.1. Passo 1 — Subir o bot em modo público e descobrir os dois IDs via `/id`

Mesmo antes de criar `config/access_control.yaml`, o bot em modo público inicia com `USER` padrão e **zero administradores** (`admins = ()`), permitindo que os dois administradores descubram seus próprios IDs com segurança:

1. Inicie os serviços no Windows 11 (`docker compose up -d --build`) e confirme `telegram_long_polling_started` em `docker compose logs --tail=40 rbac-bot`.
2. Cada um dos dois administradores abre uma conversa privada com o bot e envia `/id`.
3. O bot responderá individualmente com o Telegram User ID numérico de cada administrador.

### 4.2. Passo 2 — Criar `config/access_control.yaml` local (fora do Git)

O arquivo `config/access_control.yaml` é montado **somente leitura** (`./config:/app/config:ro`) no contêiner `rbac-bot` e permanece bloqueado no `.gitignore`:

```powershell
Set-Location "C:\Projetos\projeto-fundamentos-de-IA"
Copy-Item .\config\access_control.example.yaml .\config\access_control.yaml
git check-ignore -v config/access_control.yaml
```

Edite `config\access_control.yaml` localmente inserindo exclusivamente os dois IDs numéricos obtidos com `/id`:

```yaml
policy:
  public_access: true
  default_role: USER

roles:
  admins:
    - <ID_NUMERICO_ADMIN_1>
    - <ID_NUMERICO_ADMIN_2>
```

Reinicie apenas `rbac-bot` (sem comando destrutivo):

```powershell
docker compose restart rbac-bot
```

---

## 5. Preparação para Divulgação Pública (`@BotFather` e link `t.me`)

Para divulgar o bot com segurança:

1. No `@BotFather`, configure os comandos públicos com `/setcommands`:
   ```text
   start - Apresentacao e escopo normativo do assistente
   help - Lista de comandos e instrucoes de uso
   id - Exibe o seu proprio Telegram User ID numerico
   status - Disponibilidade do assistente e da base normativa
   rbac - Consulta direta a uma secao do RBAC sem IA
   buscar - Busca hibrida nos regulamentos aprovados
   explicar - Texto literal aprovado + explicacao fundamentada
   documentos - Lista regulamentos disponiveis na base
   versao - Versao do sistema e emendas vigentes
   ```
2. Mantenha o modo de grupos desabilitado no `@BotFather` (`/setjoingroups` → `Disable`) enquanto a política inicial for restrita a conversas privadas.
3. Divulgue **apenas** o `@username` do bot e o link público `https://t.me/<username_do_bot>`. **Nunca** divulgue o `TELEGRAM_BOT_TOKEN`.

---

## 6. Roteiro de Validação Real no Telegram (Host Windows 11)

1. Confirme `telegram_long_polling_started` nos logs de `rbac-bot` e ausência de `PTBDeprecationWarning`.
2. Com um **usuário público** (ID não listado em `roles.admins`), em conversa privada:
   - Envie `/start` → confirme `Perfil ativo: USER` (sem bloqueio).
   - Envie `/id` → confirme exibição apenas do próprio ID.
   - Envie `/status` → confirme formato público sanitizado (sem `Diagnóstico ADMIN`, heartbeat ou caminhos).
   - Envie `/rbac 43.3`, `/buscar manutencao`, `/explicar 43.3` (após aprovação/indexação) e uma pergunta fora do escopo (ex.: receita culinária) → confirme resposta fundamentada no corpus aprovado e recusa segura fora do escopo.
3. Com **cada um dos dois administradores** cadastrados em `roles.admins`:
   - Envie `/status` → confirme `--- Diagnóstico ADMIN ---` e `Administradores configurados: 2`.
   - Envie `/id` → confirme exibição exclusivamente do próprio ID.
