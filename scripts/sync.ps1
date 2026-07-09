param(
    [Parameter(Mandatory = $true)]
    [string]$OkfPath,
    [string]$ServiceUrl = "http://127.0.0.1:8765",
    [string]$TokenFile = $env:OKF_ZVEC_TOKEN_FILE
)

$ErrorActionPreference = "Stop"
if (-not $TokenFile -or -not (Test-Path -LiteralPath $TokenFile -PathType Leaf)) {
    throw "Укажите -TokenFile или задайте переменную OKF_ZVEC_TOKEN_FILE."
}

$resolvedOkf = (Resolve-Path -LiteralPath $OkfPath).Path
$temporary = Join-Path ([IO.Path]::GetTempPath()) "okf-zvec-$([guid]::NewGuid())"
$stagedOkf = Join-Path $temporary "okf"
$archive = Join-Path $temporary "okf.tgz"

try {
    New-Item -ItemType Directory -Path $stagedOkf -Force | Out-Null
    Copy-Item -Path (Join-Path $resolvedOkf "*") -Destination $stagedOkf -Recurse -Force
    tar -C $temporary -czf $archive okf

    $token = (Get-Content -Raw -LiteralPath $TokenFile).Trim()
    $json = & curl.exe -sS --fail `
        -H "X-OKF-Zvec-Token: $token" `
        -H "Content-Type: application/gzip" `
        --data-binary "@$archive" `
        "$ServiceUrl/sync"
    if ($LASTEXITCODE -ne 0 -or -not $json) {
        throw "Синхронизация завершилась ошибкой: $ServiceUrl"
    }
    $json | ConvertFrom-Json
} finally {
    Remove-Item -LiteralPath $temporary -Recurse -Force -ErrorAction SilentlyContinue
}
