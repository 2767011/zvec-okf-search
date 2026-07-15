param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Query,
    [string]$ServiceUrl = "http://127.0.0.1:8765",
    [ValidateSet("e5", "paraphrase")]
    [string]$Model = "e5",
    [ValidateSet("semantic", "fts", "fts_raw", "hybrid")]
    [string]$Mode = "hybrid",
    [ValidateRange(1, 100)]
    [int]$TopK = 5,
    [ValidateRange(0, 1)]
    [double]$MinRelevance = 0.25,
    [ValidateRange(0, 100)]
    [double]$SemanticWeight = 1,
    [ValidateRange(0, 100)]
    [double]$FtsWeight = 1,
    [string]$Type = "",
    [string]$Tags = "",
    [string]$Path = "",
    [string]$Project = "",
    [string]$DateFrom = "",
    [string]$DateTo = "",
    [ValidateSet("manual", "ai")]
    [string]$Origin = "manual",
    [string]$TokenFile = $env:OKF_ZVEC_SEARCH_TOKEN_FILE
)

$ErrorActionPreference = "Stop"
$curlArgs = @(
    "-sS", "--fail", "--get",
    "--data-urlencode", "q=$Query",
    "--data", "model=$Model",
    "--data", "mode=$Mode",
    "--data", "topk=$TopK",
    "--data", "min_relevance=$MinRelevance",
    "--data", "semantic_weight=$SemanticWeight",
    "--data", "fts_weight=$FtsWeight",
    "--data-urlencode", "type=$Type",
    "--data-urlencode", "tags=$Tags",
    "--data-urlencode", "path=$Path",
    "--data-urlencode", "project=$Project",
    "--data-urlencode", "date_from=$DateFrom",
    "--data-urlencode", "date_to=$DateTo"
)
if ($TokenFile) {
    $token = (Get-Content -Raw -LiteralPath $TokenFile).Trim()
    $curlArgs += @("--user", "okf:$token")
}
if ($Origin -eq "ai") {
    $curlArgs += @("--header", "X-OKF-Zvec-Origin: ai")
}
$curlArgs += "$ServiceUrl/search"
$json = & curl.exe @curlArgs
if ($LASTEXITCODE -ne 0 -or -not $json) {
    throw "Запрос поиска завершился ошибкой: $ServiceUrl"
}
$response = $json | ConvertFrom-Json

$response.results | Select-Object rank, relevance, score, title, path, heading, reason, text
