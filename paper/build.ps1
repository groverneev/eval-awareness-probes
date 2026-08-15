[CmdletBinding()]
param(
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$paperDirectory = $PSScriptRoot
$documentName = 'neurips_2026'
$sourceFile = "$documentName.tex"
$outputPdf = Join-Path $paperDirectory "$documentName.pdf"

function Resolve-MiKTeXTool {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $command = Get-Command "$Name.exe" -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $candidateDirectories = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64'),
        (Join-Path $env:ProgramFiles 'MiKTeX\miktex\bin\x64')
    )

    foreach ($directory in $candidateDirectories) {
        $candidate = Join-Path $directory "$Name.exe"
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "Could not find $Name.exe. Install MiKTeX, then restart VS Code."
}

function Invoke-TeXCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Host "`n==> $Description" -ForegroundColor Cyan
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

$pdfLaTeX = Resolve-MiKTeXTool -Name 'pdflatex'
$bibTeX = Resolve-MiKTeXTool -Name 'bibtex'

Push-Location $paperDirectory
try {
    $pdfLaTeXArguments = @(
        '-synctex=1',
        '-interaction=nonstopmode',
        '-file-line-error',
        '-halt-on-error',
        $sourceFile
    )

    Invoke-TeXCommand -Executable $pdfLaTeX -Arguments $pdfLaTeXArguments -Description 'Initial LaTeX pass'
    Invoke-TeXCommand -Executable $bibTeX -Arguments @($documentName) -Description 'Bibliography pass'
    Invoke-TeXCommand -Executable $pdfLaTeX -Arguments $pdfLaTeXArguments -Description 'Second LaTeX pass'
    Invoke-TeXCommand -Executable $pdfLaTeX -Arguments $pdfLaTeXArguments -Description 'Third LaTeX pass'
    Invoke-TeXCommand -Executable $pdfLaTeX -Arguments $pdfLaTeXArguments -Description 'Final stabilization pass'
}
finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $outputPdf)) {
    throw "The build completed without creating $outputPdf."
}

Write-Host "`nPaper built successfully: $outputPdf" -ForegroundColor Green

if (-not $NoOpen) {
    Start-Process -FilePath $outputPdf
}
