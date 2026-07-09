param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Query,
    [string]$ServiceUrl = "http://127.0.0.1:8765",
    [ValidateSet("e5", "paraphrase")]
    [string]$Model = "e5",
    [ValidateSet("semantic", "fts", "hybrid")]
    [string]$Mode = "hybrid",
    [ValidateRange(1, 100)]
    [int]$TopK = 5
)

$ErrorActionPreference = "Stop"
$json = & curl.exe -sS --fail --get `
    --data-urlencode "q=$Query" `
    --data "model=$Model" `
    --data "mode=$Mode" `
    --data "topk=$TopK" `
    "$ServiceUrl/search"
if ($LASTEXITCODE -ne 0 -or -not $json) {
    throw "Запрос поиска завершился ошибкой: $ServiceUrl"
}
$response = $json | ConvertFrom-Json

$response.results | Select-Object rank, score, title, path, heading, text
