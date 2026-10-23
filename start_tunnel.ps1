# Illuminathon — Skrypt automatycznego uruchamiania tunelu Cloudflare
# Pobiera cloudflared.exe jeśli go nie ma i natychmiast uruchamia tunel

$ErrorActionPreference = "Stop"

$tunnelBin = Join-Path $PSScriptRoot "cloudflared.exe"

if (-not (Test-Path $tunnelBin)) {
    Write-Host "[Illuminathon] Nie znaleziono cloudflared.exe. Pobieram oficjalne binaria..." -ForegroundColor Cyan
    $downloadUrl = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $tunnelBin -UseBasicParsing
    Write-Host "[Illuminathon] Pobrano pomyślnie!" -ForegroundColor Green
}

Write-Host "[Illuminathon] Startowanie tunelu Cloudflare na port 8080..." -ForegroundColor Yellow
Write-Host "[Wskazówka] Skopiuj wygenerowany link HTTPS (https://xxx.trycloudflare.com)" -ForegroundColor Gray
Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray

& $tunnelBin tunnel --url http://127.0.0.1:8080
