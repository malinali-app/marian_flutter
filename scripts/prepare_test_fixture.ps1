# Wrapper: convert+copy via WSL Python, then build the host Rust library.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$WslScript = "/mnt/c/Users/PierreGancel/Documents/git_malinali/marian_flutter/scripts/prepare_test_fixture_wsl.sh"

Write-Host "Running WSL convert+copy..."
wsl bash $WslScript
if ($LASTEXITCODE -ne 0) { throw "WSL prepare_test_fixture_wsl.sh failed ($LASTEXITCODE)" }

Write-Host "Building host release library..."
Push-Location (Join-Path $Root "rust")
try {
  cargo build --release
  if ($LASTEXITCODE -ne 0) { throw "cargo build --release failed" }
}
finally {
  Pop-Location
}

Write-Host "Done. Run: flutter test test/marian_service_test.dart"
