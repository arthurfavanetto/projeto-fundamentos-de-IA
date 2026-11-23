# Estado de Recuperação — Incidente Docker (22/11/2026)

> **Princípio de veracidade:** este documento registra apenas fatos comprovados no repositório Git, no GitHub e nas validações executadas. Nenhum volume antigo foi declarado recuperado nem nenhuma etapa local do Windows foi marcada como concluída sem execução real no computador do proprietário.

---

## 1. Resumo do Incidente e Decisão Definitiva

Em **22/11/2026**, foi executado indevidamente `docker compose down -v` no host Windows 11, removendo os três volumes nomeados antigos (`projeto-fundamentos-de-ia_audit_logs`, `projeto-fundamentos-de-ia_regulatory_documents`, `projeto-fundamentos-de-ia_chroma_data`).

- **Decisão definitiva do proprietário (23/11/2026):** recuperação forense dos arquivos VHDX **encerrada definitivamente**. Os arquivos físicos de backup permanecem intactos fora do fluxo; a única verificação pendente no WSL é confirmar se o UUID `f56ccfa7-68c4-4c65-b5a6-33667a8c7998` (`F:\DockerRecovery\docker_data_working.vhdx`) ainda está anexado ao `Ubuntu-24.04` e, somente se estiver, desanexá-lo explicitamente antes de abrir o Docker Desktop.
- **Conclusão operacional:** a reconstrução prossegue de forma limpa com **três volumes novos e vazios** recriados via `docker compose up -d --build` a partir da branch `main` consolidada (incluindo rotação de logs `json-file`, auditoria persistente JSONL, Bot Telegram e scripts verificáveis de backup/restauração).

---

## 2. Proteção de Backups VHDX, Volumes de Outros Projetos, Credenciais e PDF

| Ativo local (host Windows) | Localização conhecida | Política obrigatória |
| --- | --- | --- |
| VHDX original Docker | `C:\Users\Desktop\AppData\Local\Docker\wsl\disk\docker_data.vhdx` | **NÃO tocar, excluir ou reparar** |
| Backup VHDX | `D:\DockerRecovery\docker_data_recovery_2026-11-22.vhdx` | **NÃO tocar, excluir ou renomear** |
| Cópia de trabalho VHDX | `F:\DockerRecovery\docker_data_working.vhdx` | **NÃO iniciar Docker Desktop enquanto estiver anexada no WSL** |
| WSL Ubuntu forense | `Ubuntu-24.04` em `F:\WSL\Ubuntu-24.04` | **NÃO executar `wsl --unregister` nem `wsl --shutdown`** |
| Volumes de outros projetos | `n8n_data`, `ollama`, `open-webui` | **Preservar intactos — nunca reutilizar ou remover** |
| Arquivo `.env` local | `C:\Projetos\projeto-fundamentos-de-IA\.env` | **Preservar intacto (`GEMINI_API_KEY` + `TELEGRAM_BOT_TOKEN`), fora do Git** |
| Lista de acesso Telegram local | `C:\Projetos\projeto-fundamentos-de-IA\config\access_control.yaml` | **Criar a partir do template com IDs reais, fora do Git (`:ro` no Compose)** |
| PDF oficial RBAC 43 Emenda 05 | `C:\RBAC_AI_Assistant\source_pdfs\RBAC43EMD05.pdf` e `data/raw_pdfs/RBAC43EMD05.pdf` | **Preservar intacto, fora do Git** (`349196` bytes, SHA-256 `5026d2da87fc3f55c61a6b5a20dd1f50d52a1367a8550897f3864074e1a46472`) |

---

## 3. Evolução da Reconstrução no Repositório GitHub

Todas as etapas de desenvolvimento pendentes no código foram implementadas, testadas offline e integradas à `main` em PRs independentes com CI verde:

- **PR #10 (`292c116`):** Rotação de logs operacionais Docker (`json-file`, `10m` x `5`).
- **PR #11 (`5112bd5`):** Baseline de documentação pós-incidente.
- **PR #12 (`571b0b6`):** Auditoria persistente JSONL (`JsonlAuditAdapter` em `/app/logs`) + smoke test Docker no CI.
- **PR #13 (`65dc60f`):** Etapa 4 — Bot Telegram completo (Long Polling, `AccessControl` `deny_by_default`, `UserRateLimiter`, `MessageSplitter`, 8 comandos + texto livre, mount `./config:/app/config:ro`).
- **PR #14 (`ff74031`):** Etapa 5 — Avaliação de corpus oficial aprovado RBAC 43, recomendação reproduzível de calibração (mantendo `max_vector_distance=0.85` provisório) e testes de regressão normativa.
- **PR #15 (`feat/backup-restore-and-final-docs`):** Backup/restauração verificável com timestamp UTC e SHA-256 + consolidação documental final.

---

## 4. Pendências Reais no Host Windows 11 (Não Executadas pela VM Remota)

1. Verificar/liberar `F:\DockerRecovery\docker_data_working.vhdx` no WSL (somente se ainda anexado) e abrir o Docker Desktop.
2. Atualizar o clone local com `git pull --ff-only origin main` e subir os containers com `docker compose up -d --build`.
3. Validar localmente healthchecks, rotação `json-file`, persistência (ChromaDB, SQLite, auditoria JSONL) e suíte `pytest` no Windows 11.
4. Configurar `config/access_control.yaml` com Telegram User IDs reais e validar o Bot Telegram com usuário autorizado e não autorizado.
5. Ingerir `RBAC43EMD05.pdf`, revisar e aprovar páginas manualmente, indexar sob a política Free Tier (`gemini-embedding-2` 768d), rodar avaliação `--corpus official-approved` e executar o primeiro backup verificado (`scripts/backup_docker_volumes.ps1`).
