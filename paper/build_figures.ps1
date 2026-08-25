[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$paperDirectory = $PSScriptRoot
$repositoryDirectory = Split-Path $paperDirectory -Parent
$python = Join-Path $repositoryDirectory '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Could not find the project Python environment at $python."
}

$scripts = Get-ChildItem -LiteralPath $paperDirectory -Filter 'make_fig*.py' |
    Sort-Object Name
if ($scripts.Count -eq 0) {
    throw "No make_fig*.py scripts were found in $paperDirectory."
}

foreach ($script in $scripts) {
    Write-Host "`n==> Generating $($script.Name)" -ForegroundColor Cyan
    & $python $script.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "$($script.Name) failed with exit code $LASTEXITCODE."
    }
}

Write-Host "`nRan $($scripts.Count) paper figure builders; outputs are in $(Join-Path $paperDirectory 'figures')." -ForegroundColor Green
