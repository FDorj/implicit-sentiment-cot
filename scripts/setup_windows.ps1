[CmdletBinding()]
param(
    [switch]$PullModel
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RequirementsPath = Join-Path $RepoRoot "requirements.txt"
$VenvPath = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

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

$PyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
$PythonCommand = Get-Command "python" -ErrorAction SilentlyContinue
if ($PyLauncher) {
    $BootstrapExecutable = $PyLauncher.Source
    $BootstrapPrefix = @("-3")
} elseif ($PythonCommand) {
    $BootstrapExecutable = $PythonCommand.Source
    $BootstrapPrefix = @()
} else {
    throw "Python 3.10 or newer was not found. Install Python, reopen PowerShell, and rerun this script."
}

$VersionText = & $BootstrapExecutable @BootstrapPrefix -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to query the Python version."
}
$Version = [version]$VersionText.Trim()
if ($Version -lt [version]"3.10") {
    throw "Python 3.10 or newer is required; found $Version."
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating virtual environment in .venv ..."
    Invoke-Checked -Executable $BootstrapExecutable -Arguments ($BootstrapPrefix + @("-m", "venv", $VenvPath))
}

$VenvPipPackage = Join-Path $VenvPath "Lib\site-packages\pip"
if (-not (Test-Path -LiteralPath $VenvPipPackage)) {
    Write-Host "Completing pip installation in the virtual environment ..."
    Invoke-Checked -Executable $VenvPython -Arguments @("-m", "ensurepip", "--upgrade", "--default-pip")
}

Write-Host "Installing core Python dependencies ..."
Invoke-Checked -Executable $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked -Executable $VenvPython -Arguments @("-m", "pip", "install", "-r", $RequirementsPath)

$OllamaCommand = Get-Command "ollama" -ErrorAction SilentlyContinue
if ($OllamaCommand) {
    Invoke-Checked -Executable $OllamaCommand.Source -Arguments @("--version")
    if ($PullModel) {
        Write-Host "Pulling qwen3:8b with Ollama ..."
        Invoke-Checked -Executable $OllamaCommand.Source -Arguments @("pull", "qwen3:8b")
    }
} else {
    Write-Warning "Ollama was not found. Install Ollama before running language-model experiments."
    if ($PullModel) {
        throw "Cannot pull qwen3:8b because Ollama is not installed or is not on PATH."
    }
}

Write-Host "Setup complete. Python executable: $VenvPython"
Write-Host "Run the smoke test with: .\.venv\Scripts\python.exe -B test_runner.py"
