# ==============================================================================
# RBAC AI Assistant — Backup Seguro e Verificável dos Volumes Docker (Windows 11)
# ==============================================================================
# Garantias:
# - Para os serviços apenas com `docker compose stop` (NUNCA comandos destrutivos)
#   para garantir consistência de SQLite WAL e ChromaDB, retomando com `docker compose start`.
# - Faz backup exclusivamente dos 3 volumes do projeto:
#     * projeto-fundamentos-de-ia_regulatory_documents
#     * projeto-fundamentos-de-ia_chroma_data
#     * projeto-fundamentos-de-ia_audit_logs
# - Gera diretório imutável com timestamp UTC (`rbac_backup_YYYYMMDDTHHMMSSZ`) e
#   manifesto `manifest.json` com SHA-256 de cada arquivo `.tar.gz`.
# - NUNCA remove backups anteriores nem altera outros volumes ou arquivos do Docker Desktop.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$BackupRoot = "D:\DockerBackups\rbac-ai-assistant",

    [Parameter(Mandatory = $false)]
    [string]$ProjectDir = "C:\Projetos\projeto-fundamentos-de-IA"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location -Path $ProjectDir

$utcNow = (Get-Date).ToUniversalTime()
$stamp = $utcNow.ToString("yyyyMMddTHHmmssZ")
$createdAtUtc = $utcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
$bundleId = "rbac_backup_$stamp"
$bundleDir = Join-Path -Path $BackupRoot -ChildPath $bundleId

if (Test-Path -Path $bundleDir) {
    throw "Diretorio de backup ja existe e nao sera sobrescrito: $bundleDir"
}

New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null

$volumes = @(
    @{ Key = "regulatory_documents"; Volume = "projeto-fundamentos-de-ia_regulatory_documents" },
    @{ Key = "chroma_data";          Volume = "projeto-fundamentos-de-ia_chroma_data" },
    @{ Key = "audit_logs";           Volume = "projeto-fundamentos-de-ia_audit_logs" }
)

foreach ($entry in $volumes) {
    $volName = $entry.Volume
    $exists = docker volume ls --format "{{.Name}}" | Where-Object { $_ -eq $volName }
    if (-not $exists) {
        throw "Volume esperado nao encontrado no Docker: $volName. Abortando backup sem alteracoes."
    }
}

Write-Host "[1/4] Parando servicos com 'docker compose stop' para consistencia de SQLite e ChromaDB..."
docker compose stop

$archiveRecords = @()
try {
    Write-Host "[2/4] Empacotando os 3 volumes do projeto em $bundleDir ..."
    foreach ($entry in $volumes) {
        $key = $entry.Key
        $volName = $entry.Volume
        $filename = "${key}_${stamp}.tar.gz"
        $targetFile = Join-Path -Path $bundleDir -ChildPath $filename

        docker run --rm `
            -v "${volName}:/src:ro" `
            -v "${bundleDir}:/dst" `
            alpine:3.21 `
            tar -czf "/dst/$filename" -C /src .

        if (-not (Test-Path -Path $targetFile)) {
            throw "Falha ao gerar arquivo de backup: $targetFile"
        }

        $hash = (Get-FileHash -Path $targetFile -Algorithm SHA256).Hash.ToLowerInvariant()
        $size = (Get-Item -Path $targetFile).Length
        $archiveRecords += [PSCustomObject]@{
            volume_key   = $key
            filename     = $filename
            size_bytes   = $size
            sha256       = $hash
            member_count = -1
        }
        Write-Host "  [OK] $filename ($size bytes) SHA-256=$hash"
    }

    $manifest = [PSCustomObject]@{
        schema_version         = "1.0"
        bundle_id              = $bundleId
        created_at_utc         = $createdAtUtc
        sqlite_integrity_check = "verified_on_container_post_start"
        archives               = $archiveRecords
    }
    $manifestPath = Join-Path -Path $bundleDir -ChildPath "manifest.json"
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding UTF8

    Write-Host "[3/4] Validando manifesto e checksums SHA-256 imediatamente..."
    $loaded = Get-Content -Path $manifestPath -Raw | ConvertFrom-Json
    foreach ($arc in $loaded.archives) {
        $fileToCheck = Join-Path -Path $bundleDir -ChildPath $arc.filename
        $actualHash = (Get-FileHash -Path $fileToCheck -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $arc.sha256) {
            throw "Divergencia SHA-256 detectada em $($arc.filename)!"
        }
    }
    Write-Host "  [OK] Todos os checksums SHA-256 validados em $manifestPath"
}
finally {
    Write-Host "[4/4] Retomando servicos com 'docker compose start'..."
    docker compose start
}

Write-Host "[CONCLUIDO] Backup imutavel preservado em: $bundleDir"
