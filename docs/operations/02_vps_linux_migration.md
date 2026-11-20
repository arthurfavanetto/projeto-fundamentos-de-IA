# Guia de Migração Portátil — Servidor Online (VPS Linux)

A arquitetura do **RBAC AI Assistant** é 100% agnóstica ao sistema operacional do host. A migração do PC Central Windows para uma VPS Linux (Ubuntu/Debian LTS) ocorre **sem alterar nenhuma linha de código**.

---

## 1. Pré-requisitos na VPS Linux

- Ubuntu LTS ou Debian 12+ com acesso SSH por chave pública.
- **Docker Engine** e **Docker Compose Plugin** instalados.
- **Firewall UFW** configurado com política restritiva de entrada:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw enable
```

> Como o bot usa **Telegram Long Polling** (*outbound HTTPS*), **nenhuma porta de entrada HTTP/HTTPS** precisa ser aberta na VPS para o MVP.

---

## 2. Procedimento de Migração (4 Passos)

### Passo 1 — Gerar Backup no PC Central Windows
Executar o utilitário `scripts/backup_volumes.py` (implementado na Etapa 5) para gerar snapshots compactados (`.tar.gz`) dos volumes:
- `regulatory_documents` *(crítico — contém PDFs oficiais e Canonical Store SSOT)*
- `chroma_data` *(reconstruível, mas agiliza o boot)*
- `audit_logs` *(histórico de auditoria)*

### Passo 2 — Clonar o Repositório na VPS
```bash
git clone https://github.com/arthurfavanetto/projeto-fundamentos-de-IA.git rbac-ai-assistant
cd rbac-ai-assistant
```

### Passo 3 — Transferir Configurações Sensíveis e Restaurar Volumes
1. Transferir via `scp`/`sftp` (nunca via Git):
   - `.env` (ajustar `ENVIRONMENT=production`)
   - `config/access_control.yaml`
   - Arquivos `.tar.gz` de backup dos volumes
2. Restaurar os volumes Docker `regulatory_documents` e `chroma_data`.
3. **Importante:** Parar o contêiner `rbac-bot` no PC Central Windows **antes** de subir na VPS, evitando duas instâncias competindo pelo mesmo token Telegram em Long Polling.

### Passo 4 — Iniciar Operação Contínua 24/7
```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=100 rbac-bot
```

---

## 3. Checklist Pós-Migração

- [ ] `rbac-bot` e `chromadb` com status `healthy` e `restart: unless-stopped`.
- [ ] Nenhuma porta publicada no host (`docker compose ps` sem mapeamentos `0.0.0.0:*`).
- [ ] Comando `/status` e `/rbac 43.3` respondendo corretamente no Telegram.
- [ ] Logs JSONL sendo gravados no volume `audit_logs` sem exposição de segredos.
- [ ] Rotina de backup periódico configurada na VPS.
