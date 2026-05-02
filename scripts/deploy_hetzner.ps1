param(
    [string]$TargetHost = "178.104.199.55",
    [string]$User = "asad",
    [string]$RemoteDir = "/home/asad/TradeVex-Server",
    [string[]]$Files = @(
        "src/dashboard/btc_service.py",
        "tests/test_btc_service.py"
    ),
    [switch]$Restart = $true,
    [int]$WaitSeconds = 25,
    [string]$PublicApi = "https://terminal.tradevex.live/api/btc/signal?interval=15m"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Title,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Action
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Run-Step "Validating local files" {
    foreach ($relPath in $Files) {
        $localPath = Join-Path $repoRoot $relPath
        if (-not (Test-Path -LiteralPath $localPath)) {
            throw "Missing local file: $localPath"
        }
        Write-Host "OK: $localPath"
    }
}

Run-Step "Copying files to server via scp" {
    foreach ($relPath in $Files) {
        $localPath = Join-Path $repoRoot $relPath
        $remotePath = "$RemoteDir/$($relPath -replace '\\','/')"
        Write-Host "scp $relPath -> $($User)@$($TargetHost):$remotePath"
        & scp $localPath "$($User)@$($TargetHost):$remotePath"
    }
}

Run-Step "Quick server-side file check" {
    foreach ($relPath in $Files) {
        $remotePath = "$RemoteDir/$($relPath -replace '\\','/')"
        & ssh "$User@$TargetHost" "ls -l '$remotePath'"
    }
}

if ($Restart) {
    Run-Step "Restarting live process on server" {
        $restartCmd = @"
cd '$RemoteDir' ;
pkill -f '[m]ain.py --mode live' || true ;
if command -v fuser >/dev/null 2>&1; then fuser -k 8000/tcp || true ; fi ;
sleep 2 ;
nohup ./venv/bin/python main.py --mode live --config config.yaml > logs/live.out.log 2>&1 < /dev/null &
sleep $WaitSeconds ;
pgrep -af '[m]ain.py --mode live' || true ;
curl -sS http://127.0.0.1:8000/api/health || true
"@
        & ssh "$User@$TargetHost" $restartCmd
    }
}

Run-Step "Checking public API" {
    $maxRetries = 8
    $ready = $false
    for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {
        $responseRaw = & curl.exe -sS $PublicApi
        $response = [string](($responseRaw -join "`n")).Trim()
        if ($LASTEXITCODE -eq 0 -and $response.Length -gt 0 -and ($response -notmatch "(?i)502 Bad Gateway")) {
            $ready = $true
            Write-Host $response
            break
        }
        Write-Host "Public API not ready yet ($attempt/$maxRetries), retrying..." -ForegroundColor Yellow
        Start-Sleep -Seconds 4
    }
    if (-not $ready) {
        throw "Public API check failed after retries: $PublicApi"
    }
}

Write-Host ""
Write-Host "Deploy complete." -ForegroundColor Green
