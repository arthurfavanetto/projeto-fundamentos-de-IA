# Auditoria Persistente em JSONL (`JsonlAuditAdapter`)

Este documento especifica o funcionamento real da trilha de auditoria persistente do **RBAC AI Assistant** implementada via [`JsonlAuditAdapter`](../../src/rbac_assistant/infrastructure/audit/jsonl_audit_adapter.py) sob o contrato hexagonal [`AuditPort`](../../src/rbac_assistant/domain/ports/audit_port.py).

---

## 1. Separação Arquitetural (Auditoria vs. Logs Operacionais Docker)

O projeto opera **dois mecanismos independentes**, com finalidades distintas:

| Mecanismo | Localização | Formato | Rotação / Retenção | Finalidade |
| --- | --- | --- | --- | --- |
| **Logs operacionais Docker** (`structlog`) | `stdout`/`stderr` capturados pelo Docker Engine (`docker compose logs`) | JSON (`json-file` driver) | `max-size: "10m"`, `max-file: "5"` em `docker-compose.yml` | Diagnóstico técnico, inicialização, stack traces redigidos e observabilidade de container. |
| **Auditoria persistente de eventos** (`JsonlAuditAdapter`) | Volume nomeado `audit_logs` → `/app/logs/audit_events.jsonl` | JSON Lines (`.jsonl`, UTF-8) | `max_file_size_mb: 10`, `max_rotated_files: 10` em `config/settings.yaml` | Trilha durável de segurança, controle de acesso, ingestões, revisões humanas, indexações, consultas e ciclo de vida. |
| **Trilhas de revisão humana de páginas** | Volume `regulatory_documents` → SQLite `page_reviews` + `/app/data/documents/reviews/` | SQLite ACID + JSON | Preservadas integralmente (nunca rotacionadas automaticamente) | Governança regulatória de aprovação/reprovação página a página. |

---

## 2. Esquema Canônico do Evento JSONL (`schema_version: "1.0"`)

Cada linha persistida em `/app/logs/audit_events.jsonl` é um objeto JSON independente:

```json
{"schema_version":"1.0","timestamp":"2026-11-23T22:00:00.123456Z","event_type":"normative_query","outcome":"answered","actor_id":"cli","role":null,"metadata":{"query_length":48,"evidence_count":2,"blocked_count":0,"used_fallback_without_llm":false,"prompt_injection_detected":false,"citation_modifications":false,"generation_model":"gemini-4-hydrogen"}}
```

- **`timestamp`**: sempre convertido para **UTC ISO-8601** com sufixo `Z`.
- **`event_type`**: ação canônica (`system_status_check`, `service_lifecycle`, `document_ingested`, `page_review_recorded`, `document_indexed`, `normative_query`, `telegram_access_denied`, etc.).
- **`outcome`**: resultado determinístico (`healthy`, `unhealthy`, `started`, `stopped`, `success`, `failed`, `approve`, `reject`, `answered`, `refused_no_evidence`, `denied`, etc.).
- **`actor_id` / `role`**: identificador do operador/serviço (com opção `pseudonymize_user_ids: true` via hash SHA-256 truncado `usr_<16hex>`).
- **`metadata`**: exclusivamente metadados estruturados seguros.

---

## 3. Política de Redação e Privacidade

1. **Redação obrigatória e não desabilitável:** `JsonlAuditAdapter` rejeita `redact_secrets=False` e aplica [`redact_mapping`](../../src/rbac_assistant/security/secret_redactor.py) + [`redact_text`](../../src/rbac_assistant/security/secret_redactor.py) em todos os campos (`event_type`, `outcome`, `actor_id`, `role` e `metadata`), mascarando tokens Telegram, chaves Gemini (`AIza...`), tokens GitHub, AWS e chaves privadas PEM como `[REDACTED]`.
2. **Bloqueio de conteúdo sensível e documentos integrais:** chaves contendo fragmentos proibidos (`prompt`, `raw_text`, `original_text`, `document_text`, `full_text`, `pdf_bytes`, `pdf_content`, `credential`, `bearer`, `cookie`, `jwt`, `session_id`, `chat_history`) são substituídas por `[REDACTED]`.
3. **Truncamento determinístico:** strings em `metadata` são limitadas a `max_detail_string_length` (padrão `300` caracteres).

---

## 4. Durabilidade, Concorrência, Permissões e Rotação

- **Durabilidade e falha explícita:** toda gravação executa `write` + `flush()` + `os.fsync()`. Qualquer erro de disco ou permissão levanta explicitamente [`AuditWriteError`](../../src/rbac_assistant/domain/exceptions.py) — o sistema nunca declara sucesso se o evento não foi gravado.
- **Concorrência:** serialização combinada por lock de thread (por caminho canônico) + lock de arquivo entre processos (`.audit_events.jsonl.lock` usando `fcntl.flock` em Linux/POSIX e `msvcrt.locking` em Windows).
- **Compatibilidade non-root `10001:10001`:** o diretório `/app/logs` pertence ao UID/GID `10001:10001` no `Dockerfile` e os arquivos JSONL são gravados com modo `0o640` (leitura/escrita para o proprietário `appuser`, leitura para o grupo `10001`, sem acesso para outros).
- **Política de rotação e retenção:** configurada em `config/settings.yaml` (`security.audit`):
  - `filename`: `audit_events.jsonl`
  - `max_file_size_mb`: `10` (rotaciona atomicamente via `os.replace` para `.1`, `.2`, ..., `.10` ao atingir o limite)
  - `max_rotated_files`: `10` (teto aproximado de ~110 MB totais por volume `audit_logs`, evitando esgotamento de disco)

---

## 5. Validação Operacional (Docker / PowerShell)

```powershell
docker compose exec -T rbac-bot python scripts/verify_audit_persistence.py --phase write
docker compose restart rbac-bot
# Aguardar rbac-bot retornar healthy em `docker compose ps`
docker compose exec -T rbac-bot python scripts/verify_audit_persistence.py --phase verify
docker compose exec -T rbac-bot python scripts/verify_audit_persistence.py --phase cleanup
```
