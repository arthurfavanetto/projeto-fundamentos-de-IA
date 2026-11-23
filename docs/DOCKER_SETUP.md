# Guia Seguro de Setup, Reconstrução e Operação Docker

Este documento consolida a configuração real de [`Dockerfile`](../Dockerfile), [`docker/chromadb.Dockerfile`](../docker/chromadb.Dockerfile), [`docker-compose.yml`](../docker-compose.yml) e [`.dockerignore`](../.dockerignore), incluindo o procedimento seguro de liberação pós-incidente WSL/VHDX e a lista de comandos estritamente proibidos.

---

## 1. Pré-requisitos

- **Windows 11 64-bit** com **WSL2** habilitado e **Docker Desktop** (backend WSL2) — ou host Linux com Docker Engine + Docker Compose v2.
- **Python 3.12** e **`uv`** (para execução local de testes e linters fora do contêiner).
- Arquivo `.env` local na raiz do repositório (criado a partir de [`.env.example`](../.env.example), **nunca versionado no Git**).

### Configuração obrigatória do `.env` (política única Gemini Free Tier)
```env
GEMINI_GENERATION_MODEL=gemini-4-hydrogen
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMENSIONS=768
GEMINI_ALLOW_PAID_MODELS=false
CHROMA_HOST=chromadb
CHROMA_PORT=8000
ENVIRONMENT=local
LOG_LEVEL=INFO
```
- Mantenha `GEMINI_API_KEY` e `TELEGRAM_BOT_TOKEN` preenchidos apenas no `.env` local.
- **Nunca** defina `RBAC_DISABLE_DOTENV=1` nem `RBAC_BLOCK_REAL_GEMINI_API=1` no `.env` de produção (são travas exclusivas de `tests/conftest.py`).
- Mantenha o projeto Google da `GEMINI_API_KEY` **sem Cloud Billing vinculado**.

---

## 2. Pré-condição Obrigatória no Windows — Verificação de VHDX / WSL

> **NÃO inicie o Docker Desktop enquanto a cópia forense `F:\DockerRecovery\docker_data_working.vhdx` permanecer montada ou anexada como `/dev/sdd` no WSL.**

Execute no PowerShell (como Administrador) somente quando você decidir liberar o disco de investigação:

```powershell
# 1. Conferir se existe ponto de montagem ou disco anexado no Ubuntu-24.04 (somente leitura)
wsl -d Ubuntu-24.04 -- lsblk -o NAME,SIZE,RO,MOUNTPOINT,FSTYPE
wsl -d Ubuntu-24.04 -- findmnt | Select-String "sdd|DockerRecovery"

# 2. Somente com sua autorização explícita, se /dev/sdd ainda estiver montado/anexado:
# wsl -d Ubuntu-24.04 -u root -- umount /mnt/<ponto_de_montagem_se_houver>
# wsl --unmount \\.\PHYSICALDRIVE<N>   # ou: wsl --unmount "F:\DockerRecovery\docker_data_working.vhdx"
```

- **Nunca** exclua, mova, renomeie ou execute `fsck`/`e2fsck`/`mkfs` sobre:
  - `C:\Users\Desktop\AppData\Local\Docker\wsl\disk\docker_data.vhdx`
  - `D:\DockerRecovery\docker_data_recovery_2026-11-22.vhdx`
  - `F:\DockerRecovery\docker_data_working.vhdx`
- **Nunca** execute `wsl --unregister Ubuntu-24.04`.

---

## 3. Arquitetura Real do Docker Compose (`docker-compose.yml`)

| Serviço | Container | Imagem / Build | Usuário | Redes | Volumes | Limites | Healthcheck |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `chromadb` | `rbac-chromadb` | `docker/chromadb.Dockerfile` (`chromadb/chroma:1.5.9`) | `10001:10001` (non-root) | `rbac_internal_net` (`internal: true`) | `chroma_data:/data` | `1.0 CPU / 1536M` | `GET /api/v2/heartbeat` (porta `8000`, `expose` apenas, sem `ports:` no host) |
| `rbac-bot` | `rbac-bot` | `Dockerfile` (`python:3.12-slim-bookworm` + `uv 0.13.8`) | `10001:10001` (non-root) | `rbac_internal_net` + `rbac_egress_net` | `regulatory_documents:/app/data/documents`, `audit_logs:/app/logs` | `1.0 CPU / 1024M` | `python -m rbac_assistant.healthcheck` |

- **Política de reinício:** `restart: unless-stopped` em ambos os serviços.
- **Endurecimento:** `cap_drop: [ALL]` e `security_opt: [no-new-privileges:true]` em ambos os serviços.
- **Comando do `rbac-bot`:** `python -m rbac_assistant.main` (atualmente executa a fundação Etapa 1 — `CheckSystemStatusUseCase` + espera de sinal `SIGTERM`/`SIGINT`; ainda não inicia o bot Telegram).
- **Volumes de outros projetos a preservar:** `n8n_data`, `ollama`, `open-webui` (não compartilhados com este Compose).

---

## 4. Comandos Seguros de Inicialização, Diagnóstico e Operação

### 4.1. Inspeção prévia (antes de subir)
```powershell
cd C:\Projetos\projeto-fundamentos-de-IA
docker --version
docker compose version
docker volume ls
docker system df
docker compose config --quiet
docker compose config --services
```

Confirme em `docker volume ls` que `n8n_data`, `ollama` e `open-webui` permanecem preservados e que os volumes `projeto-fundamentos-de-ia_*` serão criados como volumes **novos e vazios** (se surgirem volumes desconhecidos com dados não identificados, interrompa e inspecione antes de continuar).

### 4.2. Build e inicialização segura
```powershell
docker compose up -d --build
docker compose ps -a
```

### 4.3. Healthchecks, logs e verificação não-root
```powershell
docker compose exec -T rbac-bot python -m rbac_assistant.healthcheck
docker compose exec -T rbac-bot id
docker compose exec -T chromadb id
docker compose logs --tail=50 rbac-bot
docker compose logs --tail=50 chromadb
```

### 4.4. Verificar presença de credenciais no contêiner (sem revelar valores)
```powershell
@'
import os
from rbac_assistant.config.settings import load_settings
s = load_settings()
print("TELEGRAM_BOT_TOKEN configurado no container:", bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip()))
print("GEMINI_API_KEY presente:", s.gemini_api_key is not None and bool(s.gemini_api_key.get_secret_value().strip()))
print("Modelos:", s.gemini_generation_model, "|", s.gemini_embedding_model, s.effective_embedding_dimensions)
print("Allow paid models:", s.gemini_allow_paid_models)
'@ | docker compose exec -T rbac-bot python -
```

### 4.5. Verificação segura de persistência (usando os scripts oficiais)
```powershell
# ChromaDB (escreve registro de teste, reinicia chromadb, verifica e remove apenas o registro de teste)
docker compose exec -T rbac-bot python scripts/verify_chroma_persistence.py --phase write
docker compose restart chromadb
# Aguardar rbac-chromadb retornar healthy em `docker compose ps`
docker compose exec -T rbac-bot python scripts/verify_chroma_persistence.py --phase verify
docker compose exec -T rbac-bot python scripts/verify_chroma_persistence.py --phase cleanup

# SQLite + arquivos em /app/data/documents (valida escrita, restart e remove apenas artefato de teste)
docker compose exec -T rbac-bot python scripts/verify_sqlite_persistence.py --phase write
docker compose restart rbac-bot
docker compose exec -T rbac-bot python scripts/verify_sqlite_persistence.py --phase verify
docker compose exec -T rbac-bot python scripts/verify_sqlite_persistence.py --phase cleanup
```

### 4.6. Parada e reinicialização seguras (preservando 100% dos volumes)
```powershell
# Parar sem remover volumes ou containers:
docker compose stop

# Retomar containers parados:
docker compose start

# Reiniciar um serviço específico:
docker compose restart rbac-bot
```

---

## 5. Logs, Auditoria e Rotação (Estado Real Auditado)

### 5.1. O que está implementado hoje no código
- **Logs operacionais estruturados (`src/rbac_assistant/config/logging_config.py`):**
  - Emissão JSON via `structlog` com timestamp ISO-8601 UTC (`TimeStamper(fmt="iso", utc=True)`), nível de severidade (`add_log_level`) e redação determinística de segredos (`secret_redactor.redact_mapping`).
  - Serviço principal (`python -m rbac_assistant.main`): emite em `stdout` (coletado pelo Docker Engine e consultável via `docker compose logs`).
  - CLIs (`query_assistant.py`, `index_document.py`, `evaluate_rag.py`, `ingest_regulation.py`, `inspect_document.py`): emitem logs em `stderr` (`configure_cli_logging`), mantendo `stdout` puro para `--json`.
- **Trilhas de revisão humana de páginas (Etapa 2):** persistidas duravelmente em SQLite (`page_reviews` em `/app/data/documents/catalog.sqlite3`) e arquivos em `/app/data/documents/reviews` dentro do volume `regulatory_documents`.

### 5.2. Lacunas reais identificadas na auditoria (não implementadas ainda)
- **Volume `/app/logs` (`audit_logs`) e `AuditPort`:** [`src/rbac_assistant/domain/ports/audit_port.py`](../src/rbac_assistant/domain/ports/audit_port.py) define hoje apenas o contrato `Protocol` (`AuditEvent` + `AuditPort.record()`), **sem adaptador concreto gravando arquivos JSONL em `/app/logs`**. Atualmente, `CheckSystemStatusUseCase` apenas verifica que `/app/logs` é gravável (`audit_logs_dir_writable=true`). Não marque a auditoria em arquivo como operacional até que um adaptador real (ex.: `JsonlAuditAdapter`) seja implementado na Etapa 4/5.
- **Rotação de logs do Docker:** [`docker-compose.yml`](../docker-compose.yml) ainda **não** declara bloco `logging:` explícito; utiliza a retenção padrão do Docker Desktop. Quando autorizado em PR futuro, recomenda-se adicionar em ambos os serviços (`chromadb` e `rbac-bot`):
  ```yaml
  logging:
    driver: json-file
    options:
      max-size: "10m"
      max-file: "5"
  ```

### 5.3. Consulta e teste controlado dos novos logs operacionais (sem dados sensíveis)
```powershell
# Consultar logs por serviço e janela temporal
docker compose logs --timestamps --since 15m rbac-bot
docker compose logs --timestamps --since 15m chromadb

# Conferir driver e opções de log efetivos dos containers atuais
docker inspect --format '{{.Name}} -> {{json .HostConfig.LogConfig}}' rbac-bot rbac-chromadb

# Gerar evento estruturado de teste controlado (sem credenciais) e confirmar redação
@'
import structlog
from rbac_assistant.config.logging_config import configure_logging
configure_logging(log_level="INFO", redact_secrets=True, stream="stdout")
log = structlog.get_logger("rbac_assistant.recovery_check")
log.info("recovery_log_probe", service="rbac-bot", status="ok", fake_token="123456789:AAFakeTokenNeverReal0123456789abcdef")
'@ | docker compose exec -T rbac-bot python -

# Confirmar retenção dos logs operacionais do container após restart
docker compose restart rbac-bot
docker compose logs --timestamps --tail=20 rbac-bot
```

---

## 6. Backup Seguro dos Volumes Nomeados

Antes de qualquer manutenção relevante, faça backup offline (com `docker compose stop` prévio para consistência do SQLite WAL e do ChromaDB):

```powershell
New-Item -ItemType Directory -Force -Path D:\DockerBackups\rbac-ai-assistant | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"

docker compose stop

docker run --rm -v projeto-fundamentos-de-ia_regulatory_documents:/src:ro -v D:\DockerBackups\rbac-ai-assistant:/dst alpine tar czf "/dst/regulatory_documents_$stamp.tar.gz" -C /src .
docker run --rm -v projeto-fundamentos-de-ia_chroma_data:/src:ro -v D:\DockerBackups\rbac-ai-assistant:/dst alpine tar czf "/dst/chroma_data_$stamp.tar.gz" -C /src .
docker run --rm -v projeto-fundamentos-de-ia_audit_logs:/src:ro -v D:\DockerBackups\rbac-ai-assistant:/dst alpine tar czf "/dst/audit_logs_$stamp.tar.gz" -C /src .

docker compose start
```

---

## 7. Procedimentos Estritamente Proibidos

**NUNCA execute** os comandos abaixo no projeto (risco direto de destruição definitiva dos bancos SQLite, PDFs preservados, trilhas de revisão humana e coleções ChromaDB):

- `docker compose down -v`
- `docker volume prune`
- `docker system prune` / `docker system prune -a` / `docker system prune --volumes`
- `docker volume rm ...`
- `docker rm -v ...`
- `docker builder prune`
- Evite usar `docker compose down` como procedimento rotineiro de parada — utilize sempre **`docker compose stop`**.
