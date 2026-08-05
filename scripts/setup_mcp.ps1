# Generate .cursor/mcp.json with absolute paths for this clone.
# Usage (from repo root): .\scripts\setup_mcp.ps1
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Out = Join-Path $Root ".cursor\mcp.json"

if (-not (Test-Path $VenvPy)) {
    Write-Error "Missing $VenvPy. Create a venv first:`n  python -m venv .venv`n  .\.venv\Scripts\Activate.ps1`n  pip install -r requirements.txt"
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root ".cursor") | Out-Null

$config = [ordered]@{
    mcpServers = [ordered]@{
        plan4me = [ordered]@{
            command = $VenvPy
            args    = @("-m", "mcp_server")
            cwd     = $Root
            env     = [ordered]@{
                PYTHONPATH = $Root
            }
        }
    }
}

$json = $config | ConvertTo-Json -Depth 6
# ConvertTo-Json escapes paths correctly for JSON
Set-Content -Path $Out -Value $json -Encoding utf8

Write-Host "Wrote $Out"
Write-Host "Next: reload MCP in Cursor (Settings → MCP → restart plan4me), then call health."
