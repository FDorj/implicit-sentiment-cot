[CmdletBinding()]
param(
    [switch]$SkipThesis
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$SystemPython = Get-Command "python" -ErrorAction SilentlyContinue

if (Test-Path -LiteralPath $VenvPython) {
    $Python = $VenvPython
} elseif ($SystemPython) {
    $Python = $SystemPython.Source
} else {
    throw "Python was not found. Run scripts/setup_windows.ps1 first."
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

Push-Location $RepoRoot
try {
    Write-Host "Running unit tests ..."
    Invoke-Checked -Executable $Python -Arguments @("-B", "-m", "unittest", "discover", "-s", "tests", "-v")

    Write-Host "Validating saved final pipeline results ..."
    Invoke-Checked -Executable $Python -Arguments @("-B", "experiments/run_final_pipeline.py")

    if (-not $SkipThesis) {
        & (Join-Path $PSScriptRoot "build_thesis.ps1")
        if (-not $?) {
            throw "Thesis build failed."
        }
    }
} finally {
    Pop-Location
}

Write-Host "Project verification completed successfully."
