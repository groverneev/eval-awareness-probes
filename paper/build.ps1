[CmdletBinding()]
param(
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$paperDirectory = $PSScriptRoot
$documentName = 'neurips_2026'
$sourceFile = "$documentName.tex"
$buildDirectory = Join-Path $paperDirectory '.build'
$buildPdf = Join-Path $buildDirectory "$documentName.pdf"
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

function Open-PdfInVSCode {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PdfPath
    )

    $codeCommand = Get-Command 'code.cmd' -ErrorAction SilentlyContinue
    if ($null -eq $codeCommand) {
        $codeCommand = Get-Command 'code' -ErrorAction SilentlyContinue
    }

    $codeExecutable = if ($null -ne $codeCommand) {
        $codeCommand.Source
    }
    else {
        Join-Path $env:LOCALAPPDATA 'Programs\Microsoft VS Code\bin\code.cmd'
    }

    if (-not (Test-Path -LiteralPath $codeExecutable)) {
        Write-Warning 'VS Code could not be found. Open the generated PDF from the VS Code Explorer.'
        return
    }

    & $codeExecutable --reuse-window $PdfPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'VS Code did not open the PDF. Install the recommended LaTeX Workshop extension.'
    }
}

function Update-Figures {
    $python = Join-Path (Split-Path $paperDirectory -Parent) '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Warning 'No .venv found; building with the committed figure PDFs.'
        return
    }

    $scripts = Get-ChildItem -LiteralPath $paperDirectory -Filter 'make_fig*.py' |
        Sort-Object Name
    foreach ($script in $scripts) {
        Write-Host "`n==> Regenerating $($script.Name)" -ForegroundColor Cyan
        & $python $script.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "$($script.Name) failed with exit code $LASTEXITCODE."
        }
    }
}

$pdfLaTeX = Resolve-MiKTeXTool -Name 'pdflatex'
$bibTeX = Resolve-MiKTeXTool -Name 'bibtex'

Update-Figures

New-Item -ItemType Directory -Force -Path $buildDirectory | Out-Null
Copy-Item -LiteralPath (Join-Path $paperDirectory 'references.bib') -Destination $buildDirectory -Force

Push-Location $paperDirectory
try {
    $pdfLaTeXArguments = @(
        '-synctex=1',
        '-interaction=nonstopmode',
        '-file-line-error',
        '-halt-on-error',
        "-output-directory=$buildDirectory",
        $sourceFile
    )

    Invoke-TeXCommand -Executable $pdfLaTeX -Arguments $pdfLaTeXArguments -Description 'Initial LaTeX pass'
    Push-Location $buildDirectory
    try {
        Invoke-TeXCommand -Executable $bibTeX -Arguments @($documentName) -Description 'Bibliography pass'
    }
    finally {
        Pop-Location
    }
    Invoke-TeXCommand -Executable $pdfLaTeX -Arguments $pdfLaTeXArguments -Description 'Second LaTeX pass'
    Invoke-TeXCommand -Executable $pdfLaTeX -Arguments $pdfLaTeXArguments -Description 'Third LaTeX pass'
    Invoke-TeXCommand -Executable $pdfLaTeX -Arguments $pdfLaTeXArguments -Description 'Final stabilization pass'
}
finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $buildPdf)) {
    throw "The build completed without creating $buildPdf."
}

Copy-Item -LiteralPath $buildPdf -Destination $outputPdf -Force

Write-Host "`nPaper built successfully: $outputPdf" -ForegroundColor Green

if (-not $NoOpen) {
    Open-PdfInVSCode -PdfPath $outputPdf
}
