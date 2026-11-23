# Guia Seguro de Setup, Reconstrução e Operação Docker

Este documento consolida a configuração real de [`Dockerfile`](../Dockerfile), [`docker/chromadb.Dockerfile`](../docker/chromadb.Dockerfile), [`docker-compose.yml`](../docker-compose.yml) e [`.dockerignore`](../.dockerignore), incluindo rotação de logs operacionais, auditoria persistente em `/app/logs`, montagem somente leitura de `config/access_control.yaml`, backup verificável e procedimentos proibidos.

---

## 1. Pré-requisitos

- **Windows 11 64-bit** com **WSL2** habilitado e **Docker Desktop** (backend WSL2) — ou host Linux com Docker Engine + Docker Compose v2.
- **Python 3.12** e **`uv`** (para execução local de testes, linters e verificação de backups fora do contêiner).
- Arquivo `.env` local na raiz do repositório (a partir de [`.env.example`](../.env.example), **nunca versionado no Git**).
- Arquivo `config/access_control.yaml` local (a partir de [`config/access_control.example.yaml`](../config/access_control.example.yaml), **nunca versionado no Git**) com os Telegram User IDs numéricos autorizados.

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
- **Nunca** defina `RBAC_DISABLE_DOTENV=1`, `RBAC_BLOCK_REAL_GEMINI_API=1` ou `RBAC_BLOCK_REAL_TELEGRAM_API=1` no `.env` operacional (são travas exclusivas de `tests/conftest.py`).
- Mantenha o projeto Google da `GEMINI_API_KEY` **sem Cloud Billing vinculado**.

---

## 2. Pré-condição Obrigatória no Windows — Verificação de VHDX / WSL

> **NÃO inicie o Docker Desktop enquanto a cópia forense `F:\DockerRecovery\docker_data_working.vhdx` (UUID `f56ccfa7-68c4-4c65-b5a6-33667a8c7998`) permanecer montada ou anexada no WSL.**

Execute no PowerShell (como Administrador):

```powershell
# 1. Conferir se o UUID f56ccfa7-68c4-4c65-b5a6-33667a8c7998 ainda aparece no Ubuntu-24.04
wsl -d Ubuntu-24.04 -- lsblk -o NAME,UUID,SIZE,RO,MOUNTPOINT,FSTYPE

# 2. Somente se o UUID aparecer na saída acima:
#    a) Se houver ponto de montagem ativo: wsl -d Ubuntu-24.04 -u root -- umount "<MOUNTPOINT>"
#    b) Desanexar ESPECIFICAMENTE a cópia de trabalho (NUNCA `wsl --unmount` sem caminho!):
#       wsl --unmount "F:\DockerRecovery\docker_data_working.vhdx"
```

- **Nunca** exclua, mova, renomeie ou execute `fsck`/`e2fsck`/`mkfs` sobre `C:\Users\Desktop\AppData\Local\Docker\wsl\disk\docker_data.vhdx`, `D:\DockerRecovery\docker_data_recovery_2026-11-22.vhdx` ou `F:\DockerRecovery\docker_data_working.vhdx`.
- **Nunca** execute `wsl --unregister Ubuntu-24.04` nem `wsl --shutdown` para liberar o disco.

---

## 3. Arquitetura Real do Docker Compose (`docker-compose.yml`)

| Serviço | Container | Imagem / Build | Usuário | Redes | Volumes / Mounts | Rotação Logs Docker | Limites | Healthcheck |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `chromadb` | `rbac-chromadb` | `docker/chromadb.Dockerfile` (`chromadb/chroma:1.5.9`) | `10001:10001` | `rbac_internal_net` (`internal: true`) | `chroma_data:/data` | `json-file` (`10m` x `5`) | `1.0 CPU / 1536M` | `GET /api/v2/heartbeat` (porta `8000` interna, sem `ports:`) |
| `rbac-bot` | `rbac-bot` | `Dockerfile` (`python:3.12-slim-bookworm` + `uv 0.13.8`) | `10001:10001` | `rbac_internal_net` + `rbac_egress_net` | `regulatory_documents:/app/data/documents`, `audit_logs:/app/logs`, `./config:/app/config:ro` | `json-file` (`10m` x `5`) | `1.0 CPU / 1024M` | `python -m rbac_assistant.healthcheck` |

- **Política de reinício:** `restart: unless-stopped` em ambos os serviços.
- **Endurecimento:** `cap_drop: [ALL]` e `security_opt: [no-new-privileges:true]` em ambos os serviços.
- **Comando do `rbac-bot`:** `python -m rbac_assistant.main` (Composition Root completo com auditoria JSONL em `/app/logs`, SQLite SSOT, ChromaDB, RAG e Bot Telegram em Long Polling — opera automaticamente em modo standby seguro quando `TELEGRAM_BOT_TOKEN` está ausente/placeholder).
- **Volumes de outros projetos a preservar:** `n8n_data`, `ollama`, `open-webui`.

---

## 4. Comandos Seguros de Inicialização, Diagnóstico e Persistência (PowerShell)

### 4.1. Inspeção prévia e subida
```powershell
Set-Location "C:\Projetos\projeto-fundamentos-de-IA"
docker --version
docker compose version
docker volume ls
docker compose config --quiet
docker compose up -d --build
docker compose ps -a
```

### 4.2. Healthchecks, usuário não-root e rotação de logs efetiva
```powershell
docker compose exec -T rbac-bot python -m rbac_assistant.healthcheck
docker compose exec -T rbac-bot id
docker compose exec -T chromadb id
docker inspect --format "{{.Name}} -> {{json .HostConfig.LogConfig}}" rbac-bot rbac-chromadb
docker compose logs --timestamps --tail=50 rbac-bot
docker compose logs --timestamps --tail=50 chromadb
```

### 4.3. Verificação de persistência (ChromaDB, SQLite e Auditoria JSONL pós-restart)
```powershell
# 1. ChromaDB
docker compose exec -T rbac-bot python scripts/verify_chroma_persistence.py --phase write
docker compose restart chromadb
Start-Sleep -Seconds 15
docker compose exec -T rbac-bot python scripts/verify_chroma_persistence.py --phase verify
docker compose exec -T rbac-bot python scripts/verify_chroma_persistence.py --phase cleanup

# 2. SQLite + PDF + Revisão + Auditoria JSONL (/app/logs)
docker compose exec -T rbac-bot python scripts/verify_sqlite_persistence.py --phase write
docker compose exec -T rbac-bot python scripts/verify_audit_persistence.py --phase write
docker compose restart rbac-bot
Start-Sleep -Seconds 10
docker compose exec -T rbac-bot python scripts/verify_sqlite_persistence.py --phase verify
docker compose exec -T rbac-bot python scripts/verify_audit_persistence.py --phase verify
docker compose exec -T rbac-bot python scripts/verify_sqlite_persistence.py --phase cleanup
docker compose exec -T rbac-bot python scripts/verify_audit_persistence.py --phase cleanup
```

### 4.4. Parada e reinicialização seguras (preservando 100% dos volumes)
```powershell
docker compose stop
docker compose start
docker compose restart rbac-bot
```

---

## 5. Logs Operacionais vs. Auditoria Persistente (Estado Real Atualizado)

1. **Rotação de logs operacionais Docker (ativa desde PR #10):** `docker-compose.yml` configura `logging: {driver: json-file, options: {max-size: "10m", max-file: "5"}}` em `chromadb` e `rbac-bot`, validado em CI Docker real.
2. **Auditoria persistente em `/app/logs` (ativa desde PR #12):** [`JsonlAuditAdapter`](../src/rbac_assistant/infrastructure/audit/jsonl_audit_adapter.py) grava eventos estruturados UTC ISO-8601 em `/app/logs/audit_events.jsonl` com redação obrigatória de segredos, concorrência thread/processo, `fsync`, falha explícita via `AuditWriteError` e rotação independente por tamanho (`10 MB` x `10` arquivos). Detalhes em [`docs/operations/AUDIT_LOGGING.md`](./operations/AUDIT_LOGGING.md).
3. **Trilhas de revisão humana de páginas:** continuam persistidas em SQLite (`page_reviews`) e `/app/data/documents/reviews/` dentro do volume `regulatory_documents`.

---

## 6. Backup e Restauração Seguros

Utilize os scripts verificáveis com timestamp UTC e manifesto SHA-256 documentados em [`docs/operations/BACKUP_AND_RESTORE.md`](./operations/BACKUP_AND_RESTORE.md):

```powershell
.\scripts\backup_docker_volumes.ps1 -BackupRoot "D:\DockerBackups\rbac-ai-assistant"
```

---

## 7. Procedimentos Estritamente Proibidos

**NUNCA execute** os comandos abaixo no projeto:

- `docker compose down -v`
- `docker volume prune`
- `docker system prune` / `docker system prune -a` / `docker system prune --volumes`
- `docker volume rm ...`
- `docker rm -v ...`
- `docker builder prune`
- Evite usar `docker compose down` rotineiramente — utilize sempre **`docker compose stop`**.
