# ==============================================================================
# RBAC AI Assistant — Restauracao Segura e Verificavel dos Volumes Docker (Windows 11)
# ==============================================================================
# Garantias:
# - Exige `-ConfirmRestore` explicito.
# - Valida integralmente `manifest.json` e todos os checksums SHA-256 ANTES de
#   parar os servicos ou tocar em qualquer volume.
# - Opera exclusivamente nos 3 volumes do projeto (`projeto-fundamentos-de-ia_*`).
# - NUNCA executa comandos destrutivos de limpeza Docker nem modifica arquivos do host.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BundleDir,

    [Parameter(Mandatory = $false)]
    [string]$ProjectDir = "C:\Projetos\projeto-fundamentos-de-IA",

    [Parameter(Mandatory = $false)]
    [switch]$ConfirmRestore
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $ConfirmRestore) {
    throw "Restauracao bloqueada por seguranca. Reexecute informando explicitamente -ConfirmRestore."
}

$manifestPath = Join-Path -Path $BundleDir -ChildPath "manifest.json"
if (-not (Test-Path -Path $manifestPath)) {
    throw "Manifesto nao encontrado em: $manifestPath"
}

Write-Host "[1/4] Validando manifesto e checksums SHA-256 antes de qualquer alteracao..."
$manifest = Get-Content -Path $manifestPath -Raw | ConvertFrom-Json
$volumeMap = @{
    "regulatory_documents" = "projeto-fundamentos-de-ia_regulatory_documents"
    "chroma_data"          = "projeto-fundamentos-de-ia_chroma_data"
    "audit_logs"           = "projeto-fundamentos-de-ia_audit_logs"
}

foreach ($requiredKey in $volumeMap.Keys) {
    $match = $manifest.archives | Where-Object { $_.volume_key -eq $requiredKey }
    if (-not $match) {
        throw "Volume obrigatorio ausente no manifesto: $requiredKey"
    }
    $archivePath = Join-Path -Path $BundleDir -ChildPath $match.filename
    if (-not (Test-Path -Path $archivePath)) {
        throw "Arquivo de backup ausente: $archivePath"
    }
    $actualHash = (Get-FileHash -Path $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $match.sha256.ToLowerInvariant()) {
        throw "SHA-256 divergente em $($match.filename)! Restauracao abortada sem alteracoes."
    }
    Write-Host "  [OK] $($match.filename) verificado (SHA-256 intacto)."
}

Set-Location -Path $ProjectDir
Write-Host "[2/4] Parando servicos com 'docker compose stop'..."
docker compose stop

try {
    Write-Host "[3/4] Restaurando os 3 volumes do projeto a partir de $BundleDir ..."
    foreach ($arc in $manifest.archives) {
        $key = $arc.volume_key
        if (-not $volumeMap.ContainsKey($key)) {
            continue
        }
        $volName = $volumeMap[$key]
        $filename = $arc.filename

        docker run --rm `
            -v "${volName}:/dst" `
            -v "${BundleDir}:/src:ro" `
            alpine:3.21 `
            sh -c "find /dst -mindepth 1 -maxdepth 1 -exec rm -rf {} + && tar -xzf /src/$filename -C /dst && chown -R 10001:10001 /dst"

        Write-Host "  [OK] Volume $volName restaurado a partir de $filename."
    }
}
finally {
    Write-Host "[4/4] Retomando servicos com 'docker compose start' e executando verificacao pos-restauracao..."
    docker compose start
}

Start-Sleep -Seconds 15
docker compose ps -a
docker compose exec -T rbac-bot python -m rbac_assistant.healthcheck
docker compose exec -T rbac-bot python scripts/inspect_document.py
docker compose exec -T rbac-bot python scripts/index_document.py --status
Write-Host "[CONCLUIDO] Restauracao e verificacoes pos-restauracao finalizadas."
