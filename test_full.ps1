# Full test script
param(
    [string]$VideoPath = "F:\downloads\The_AI-Native_Reconstruct.mp4",
    [string]$BaseUrl   = "http://localhost:5000",
    [string]$Voice     = "edge_yuyang"
)

function Upload-Video($FilePath) {
    $client = New-Object System.Net.Http.HttpClient
    $content = New-Object System.Net.Http.MultipartFormDataContent
    $fileStream = [System.IO.File]::OpenRead($FilePath)
    $fileContent = New-Object System.Net.Http.StreamContent($fileStream)
    $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("video/mp4")
    $content.Add($fileContent, "video", [System.IO.Path]::GetFileName($FilePath))
    $resp = $client.PostAsync("$BaseUrl/api/upload", $content).Result
    $body = $resp.Content.ReadAsStringAsync().Result
    $fileStream.Close()
    $client.Dispose()
    return ($body | ConvertFrom-Json).job_id
}

function Poll-Job($JobId, $MaxSecs=300) {
    $start = Get-Date
    while ((Get-Date) - $start -lt [TimeSpan]::FromSeconds($MaxSecs)) {
        try {
            $r = Invoke-RestMethod "$BaseUrl/api/job_info/$JobId"
            Write-Host "$(Get-Date -f HH:mm:ss) [$($r.progress)%] $($r.status) — $($r.message)"
            if ($r.status -eq "completed") { return $r }
            if ($r.status -eq "error")     { Write-Host "ERROR: $($r.message)"; return $null }
        } catch { Write-Host "$(Get-Date -f HH:mm:ss) waiting..." }
        Start-Sleep 5
    }
    return $null
}

# 1. Upload & generate subtitles
Write-Host "`n=== Subtitle Generation ===" -ForegroundColor Cyan
$jobId = Upload-Video $VideoPath
Write-Host "Job: $jobId"

$result = Poll-Job $jobId 300
if (-not $result) { Write-Host "FAILED"; exit 1 }

$subs = $result.subtitles
Write-Host "Generated: $($subs.Count) subtitles" -ForegroundColor Green
$subs[0..2] | ForEach-Object {
    $t = $_.translation; if ($t.Length -gt 50) { $t = $t.Substring(0,50) + "..." }
    Write-Host "  $($_.start)s-$($_.end)s: $t"
}

# 2. Dubbing
Write-Host "`n=== Dubbing ($Voice) ===" -ForegroundColor Cyan
$body = @{voice=$Voice} | ConvertTo-Json
Invoke-RestMethod -Uri "$BaseUrl/api/dub/$jobId" -Method POST -Body $body -ContentType "application/json" | Out-Null

$timeout = (Get-Date).AddSeconds(300)
while ((Get-Date) -lt $timeout) {
    Start-Sleep 5
    $dubbed = "C:\Users\tanga\video-subtitle\uploads\${jobId}_dubbed.mp4"
    if (Test-Path $dubbed) {
        $mb = [math]::Round((Get-Item $dubbed).Length/1MB, 1)
        Write-Host "dubbed.mp4: $mb MB" -ForegroundColor Green
        break
    }
    $segs = (Get-ChildItem "C:\Users\tanga\video-subtitle\uploads\${jobId}_audio\seg_*.wav" -EA SilentlyContinue).Count
    Write-Host "$(Get-Date -f HH:mm:ss) TTS segments: $segs / $($subs.Count)"
}

Write-Host "`n=== All tests done ===" -ForegroundColor Yellow
Write-Host "Job ID for browser: $jobId"
