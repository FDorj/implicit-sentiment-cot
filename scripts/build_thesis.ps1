[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ThesisDirectoryName = "قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir"
$ThesisDirectory = Join-Path $RepoRoot $ThesisDirectoryName
$ThesisSource = Join-Path $ThesisDirectory "AUTthesis.tex"
$ThesisPdf = Join-Path $ThesisDirectory "AUTthesis.pdf"
$ThesisLog = Join-Path $ThesisDirectory "AUTthesis.log"

function Require-Command {
    param([Parameter(Mandatory = $true)][string]$Name)

    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $Command) {
        throw "$Name was not found. Install MiKTeX and make sure its bin directory is on PATH."
    }
    return $Command.Source
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $ThesisSource)) {
    throw "Thesis source not found: $ThesisSource"
}

$XeLaTeX = Require-Command "xelatex"
$BibTeX = Require-Command "bibtex"
$XeLaTeXArguments = @(
    "-interaction=nonstopmode",
    "-halt-on-error",
    "-file-line-error",
    "AUTthesis.tex"
)

Push-Location $ThesisDirectory
try {
    Write-Host "XeLaTeX pass 1/3 ..."
    Invoke-Checked -Executable $XeLaTeX -Arguments $XeLaTeXArguments
    Write-Host "BibTeX pass ..."
    Invoke-Checked -Executable $BibTeX -Arguments @("AUTthesis")
    Write-Host "XeLaTeX pass 2/3 ..."
    Invoke-Checked -Executable $XeLaTeX -Arguments $XeLaTeXArguments
    Write-Host "XeLaTeX pass 3/3 ..."
    Invoke-Checked -Executable $XeLaTeX -Arguments $XeLaTeXArguments
} finally {
    Pop-Location
}

if (-not (Test-Path -LiteralPath $ThesisPdf)) {
    throw "Thesis build completed without creating AUTthesis.pdf."
}
if (-not (Test-Path -LiteralPath $ThesisLog)) {
    throw "Thesis build did not create AUTthesis.log."
}

$LogText = Get-Content -Raw -LiteralPath $ThesisLog
$FatalPatterns = @(
    "! LaTeX Error",
    "Undefined control sequence",
    "Emergency stop",
    "Fatal error occurred"
)
foreach ($Pattern in $FatalPatterns) {
    if ($LogText.Contains($Pattern)) {
        throw "LaTeX log contains a fatal error marker: $Pattern"
    }
}

$UnresolvedPattern = "There were undefined references|Citation .+ undefined|Reference .+ undefined"
if ($LogText -match $UnresolvedPattern) {
    throw "LaTeX log contains unresolved citations or references."
}
if ($LogText -match "Overfull \\hbox") {
    throw "LaTeX log contains an overfull hbox."
}

$OverfullVboxCount = ([regex]::Matches($LogText, "Overfull \\vbox")).Count
if ($OverfullVboxCount -gt 0) {
    Write-Warning "LaTeX reported $OverfullVboxCount overfull vbox warning(s). These are non-fatal and currently come from the empty chapter skeletons."
}

$PdfInfo = Get-Item -LiteralPath $ThesisPdf
if ($PdfInfo.Length -le 0) {
    throw "AUTthesis.pdf is empty."
}

Write-Host "Thesis PDF created successfully: $($PdfInfo.FullName) ($($PdfInfo.Length) bytes)"
