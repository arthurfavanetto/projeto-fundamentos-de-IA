# Procedimento Operacional Seguro — Backup, Verificação SHA-256 e Restauração

Este guia especifica a rotina de backup e restauração dos **três novos volumes nomeados** do **RBAC AI Assistant** no Windows 11 (Docker Desktop) e em ambientes Python/Linux.

> **Estado de validação:** o motor Python ([`backup_manager.py`](../../src/rbac_assistant/infrastructure/storage/backup_manager.py)), a CLI [`scripts/backup_restore.py`](../../scripts/backup_restore.py) e os contratos de segurança dos scripts PowerShell foram validados **100% offline na VM Linux e no CI GitHub Actions**. A execução real contra os volumes Docker locais do Windows 11 permanece **PENDENTE DE VALIDAÇÃO LOCAL NO WINDOWS**.

---

## 1. Escopo e Garantias de Segurança

| Componente | Volume Docker | Conteúdo preservado |
| --- | --- | --- |
| Documentos e SQLite SSOT | `projeto-fundamentos-de-ia_regulatory_documents` | `catalog.sqlite3` (catálogo, páginas, unidades, FTS5, `page_reviews`), `raw_pdfs/`, `extracted/`, `reviews/` |
| Índice Vetorial | `projeto-fundamentos-de-ia_chroma_data` | Coleções ChromaDB 1.5.9 (`regulatory_chunks_ge2_768_v1`) |
| Auditoria Persistente | `projeto-fundamentos-de-ia_audit_logs` | `audit_events.jsonl` e arquivos rotacionados `.jsonl.1`..`.10` |

### Garantias obrigatórias
1. **Consistência transacional:** os scripts PowerShell param os containers apenas com `docker compose stop` antes de empacotar SQLite WAL e ChromaDB, retomando sempre via `docker compose start` no bloco `finally`. O motor Python usa `sqlite3.Connection.backup()` + `PRAGMA integrity_check`.
2. **Timestamp UTC e imutabilidade:** cada backup cria um diretório dedicado `rbac_backup_YYYYMMDDTHHMMSSZ` (timestamp UTC) e **nunca sobrescreve nem apaga backups anteriores automaticamente**.
3. **Integridade criptográfica:** gera `manifest.json` com `size_bytes` e **SHA-256** de cada `.tar.gz` e revalida os hashes imediatamente após o backup e **antes** de qualquer restauração.
4. **Isolamento absoluto:** jamais toca nos volumes `n8n_data`, `ollama`, `open-webui`, nos VHDX do Docker Desktop ou em comandos destrutivos (`down -v`, `prune`, `volume rm`, `rm -v`).

---

## 2. Localização e Política de Retenção Recomendada

- **Diretório padrão no Windows 11:** `D:\DockerBackups\rbac-ai-assistant\` (unidade separada do disco `C:` do sistema e dos VHDX do Docker Desktop).
- **Retenção manual recomendada:** manter ao menos os últimos `7` backups diários e `1` backup antes de cada ingestão, aprovação em lote ou atualização de versão. Qualquer descarte de backups antigos deve ser **decisão manual explícita do proprietário** — nenhum script apaga backups anteriores automaticamente.

---

## 3. Execução no Windows 11 (PowerShell)

### 3.1. Criar backup completo e validar SHA-256
```powershell
Set-Location "C:\Projetos\projeto-fundamentos-de-IA"
.\scripts\backup_docker_volumes.ps1 -BackupRoot "D:\DockerBackups\rbac-ai-assistant"
```

### 3.2. Validar um bundle existente a qualquer momento (somente leitura)
```powershell
uv run python scripts/backup_restore.py verify --bundle-dir "D:\DockerBackups\rbac-ai-assistant\rbac_backup_<YYYYMMDDTHHMMSSZ>"
```

### 3.3. Restaurar um bundle verificado (exige `-ConfirmRestore` explícito)
```powershell
Set-Location "C:\Projetos\projeto-fundamentos-de-IA"
.\scripts\restore_docker_volumes.ps1 `
    -BundleDir "D:\DockerBackups\rbac-ai-assistant\rbac_backup_<YYYYMMDDTHHMMSSZ>" `
    -ConfirmRestore
```

### 3.4. Teste de recuperação de falhas (Drill não-destrutivo)
Para testar a restauração sem impactar os volumes de produção, valide a integridade e extraia o bundle para um diretório temporário isolado usando a CLI Python:
```powershell
uv run python scripts/backup_restore.py verify --bundle-dir "D:\DockerBackups\rbac-ai-assistant\rbac_backup_<YYYYMMDDTHHMMSSZ>" --json
uv run python scripts/backup_restore.py restore `
    --bundle-dir "D:\DockerBackups\rbac-ai-assistant\rbac_backup_<YYYYMMDDTHHMMSSZ>" `
    --documents-dir "$env:TEMP\rbac_restore_drill\documents" `
    --chroma-dir "$env:TEMP\rbac_restore_drill\chroma" `
    --audit-logs-dir "$env:TEMP\rbac_restore_drill\logs" `
    --confirm-restore --json
```
Confirme `"restored": true` e `"sqlite_integrity_check": "ok"`.
