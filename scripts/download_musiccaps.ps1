[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)]
    [int]$Limit = 0,
    [switch]$FastResume,
    [switch]$RetryUnavailable,
    [string]$MetadataCsv = 'data/raw/musiccaps_official.csv',
    [string]$AudioDir = 'data/raw/musiccaps_audio'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$invariant = [Globalization.CultureInfo]::InvariantCulture

function Get-ProjectPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

function Find-Executable([string]$Name) {
    $command = Get-Command "$Name.exe" -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($command) { return $command.Source }
    $candidate = Join-Path $env:LOCALAPPDATA "Microsoft/WinGet/Links/$Name.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    # Portable WinGet installs may update persistent PATH without creating Links.
    $persistentPaths = @(
        [Environment]::GetEnvironmentVariable('Path', 'User'),
        [Environment]::GetEnvironmentVariable('Path', 'Machine')
    )
    foreach ($entry in (($persistentPaths -join ';') -split ';')) {
        if ([string]::IsNullOrWhiteSpace($entry)) { continue }
        $directory = [Environment]::ExpandEnvironmentVariables($entry.Trim().Trim('"'))
        $candidate = Join-Path $directory "$Name.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    return $null
}

function Get-ClipValidation([string]$Path, [double]$ExpectedSeconds) {
    $probeText = & $ffprobe -v error -show_entries `
        'format=duration,format_name:stream=codec_type,sample_rate,channels' -of json $Path 2>$null
    if ($LASTEXITCODE -ne 0) { return 'ffprobe could not read the audio' }
    try {
        $probe = ($probeText -join "`n") | ConvertFrom-Json
        $audioStreams = @($probe.streams | Where-Object { $_.codec_type -eq 'audio' })
        if ($probe.format.format_name -ne 'wav' -or $audioStreams.Count -ne 1 -or
            [int]$audioStreams[0].sample_rate -le 0 -or [int]$audioStreams[0].channels -le 0) {
            return 'expected a WAV file containing one valid audio stream'
        }
        $seconds = [double]::Parse([string]$probe.format.duration, $invariant)
        if ([double]::IsNaN($seconds) -or [double]::IsInfinity($seconds) -or
            [Math]::Abs($seconds - $ExpectedSeconds) -gt 0.02) {
            return "duration is $seconds seconds; expected $ExpectedSeconds seconds (+/- 0.02)"
        }
    }
    catch { return "invalid audio metadata: $($_.Exception.Message)" }
    # Decode the complete file so a plausible header cannot hide corrupt audio.
    & $ffmpeg -hide_banner -v error -xerror -i $Path -map '0:a:0' -f null - 2>$null
    if ($LASTEXITCODE -ne 0) { return 'FFmpeg could not decode the complete audio file' }
    return $null
}

$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python is missing: $python. Create the project .venv first."
}
$metadataPath = Get-ProjectPath $MetadataCsv
$audioPath = Get-ProjectPath $AudioDir
if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
    throw "Metadata CSV not found: $metadataPath"
}
$ffmpeg = Find-Executable 'ffmpeg'
$ffprobe = Find-Executable 'ffprobe'
if (-not $ffmpeg -or -not $ffprobe) {
    throw 'FFmpeg/ffprobe not found. Run: winget install --exact --id Gyan.FFmpeg --source winget. Then restart VS Code, or ensure both executables are in your per-user WinGet Links folder.'
}
& $ffmpeg -version *> $null
if ($LASTEXITCODE -ne 0) { throw "FFmpeg could not start: $ffmpeg" }
& $ffprobe -version *> $null
if ($LASTEXITCODE -ne 0) { throw "ffprobe could not start: $ffprobe" }
$ffmpegDirectory = Split-Path -Parent $ffmpeg
if (-not (Test-Path -LiteralPath (Join-Path $ffmpegDirectory 'ffprobe.exe') -PathType Leaf)) {
    throw "Place ffmpeg.exe and ffprobe.exe in the same directory: $ffmpegDirectory"
}

$runtime = $null
$deno = Find-Executable 'deno'
if ($deno) {
    $versionText = (& $deno --version 2>$null) -join ' '
    if ($LASTEXITCODE -eq 0 -and $versionText -match 'deno\s+(\d+)\.' -and [int]$Matches[1] -ge 2) {
        $runtime = "deno:$deno"
    }
}
if (-not $runtime) {
    $node = Find-Executable 'node'
    if ($node) {
        $versionText = (& $node --version 2>$null) -join ' '
        if ($LASTEXITCODE -eq 0 -and $versionText -match '^v(\d+)\.' -and [int]$Matches[1] -ge 22) {
            $runtime = "node:$node"
        }
    }
}
if (-not $runtime) {
    throw 'No supported JavaScript runtime found (Deno >=2 or Node >=22). Run: winget install --exact --id DenoLand.Deno --source winget. Then restart VS Code.'
}
& $python -m yt_dlp --version
if ($LASTEXITCODE -ne 0) {
    throw 'yt-dlp is missing from the project environment. Run: .\.venv\Scripts\python.exe -m pip install --upgrade "yt-dlp[default]"'
}

$rows = @(Import-Csv -LiteralPath $metadataPath)
if ($rows.Count -eq 0) { throw 'Metadata CSV has no rows.' }
foreach ($required in @('ytid', 'start_s', 'end_s')) {
    if ($required -notin $rows[0].PSObject.Properties.Name) { throw "CSV is missing column: $required" }
}
if ($Limit -gt 0) { $rows = @($rows | Select-Object -First $Limit) }
# Validate every selected row before starting any downloads.
$clips = @(foreach ($row in $rows) {
    if ($row.ytid -notmatch '^[A-Za-z0-9_-]{11}$') { throw "Invalid video ID: $($row.ytid)" }
    $start = [double]::Parse($row.start_s, $invariant)
    $end = [double]::Parse($row.end_s, $invariant)
    if ([double]::IsNaN($start) -or [double]::IsInfinity($start) -or
        [double]::IsNaN($end) -or [double]::IsInfinity($end) -or $start -lt 0 -or $end -le $start) {
        throw "Invalid interval for $($row.ytid): $start - $end"
    }
    $startText = $start.ToString('0.################', $invariant)
    $endText = $end.ToString('0.################', $invariant)
    [pscustomobject]@{
        Id = $row.ytid
        Name = "$($row.ytid)_${startText}_${endText}"
        Section = "*${startText}-${endText}"
        Duration = $end - $start
    }
})

New-Item -ItemType Directory -Force -Path $audioPath | Out-Null
$logDirectory = Join-Path $projectRoot 'results/task2'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$failureLog = Join-Path $logDirectory 'audio_download_failures.txt'
$unavailableLog = Join-Path $logDirectory 'audio_unavailable.jsonl'
$unavailableIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
if (-not $RetryUnavailable -and (Test-Path -LiteralPath $unavailableLog -PathType Leaf)) {
    foreach ($line in [IO.File]::ReadLines($unavailableLog)) {
        if (-not [string]::IsNullOrWhiteSpace($line)) {
            $entry = $line | ConvertFrom-Json
            [void]$unavailableIds.Add([string]$entry.sample_id)
        }
    }
}
$unavailable = 0
$downloadRunner = Join-Path $PSScriptRoot 'run_audio_download.py'
$failures = 0
$downloaded = 0
$skipped = 0
Write-Host "FFmpeg: $ffmpeg"
Write-Host "JavaScript runtime: $runtime"
Write-Host "Audio directory: $audioPath"

# Read the old loop's archive once, rather than launching yt-dlp for each ID.
$archivedIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$archivePath = Join-Path $logDirectory 'audio_downloaded.txt'
if ($FastResume -and (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    foreach ($line in [IO.File]::ReadLines($archivePath)) {
        if ($line -match '^youtube\s+([A-Za-z0-9_-]{11})\s*$') {
            [void]$archivedIds.Add($Matches[1])
        }
    }
}
$fastSkipped = 0
if ($FastResume) {
    Write-Host "Fast resume: $($archivedIds.Count) archived IDs loaded. Existing archived WAVs will skip decoding checks."
}

foreach ($clip in $clips) {
    $wavPath = Join-Path $audioPath "$($clip.Name).wav"
    if ($unavailableIds.Contains($clip.Name) -and -not (Test-Path -LiteralPath $wavPath)) {
        $unavailable++
        continue
    }
    if ($FastResume -and $archivedIds.Contains($clip.Id) -and
        (Test-Path -LiteralPath $wavPath -PathType Leaf) -and
        (Get-Item -LiteralPath $wavPath).Length -gt 44) {
        $skipped++
        $fastSkipped++
        continue
    }
    if (Test-Path -LiteralPath $wavPath) {
        $issue = Get-ClipValidation $wavPath $clip.Duration
        if ($issue) {
            Add-Content -LiteralPath $failureLog -Value "$(Get-Date -Format o)`t$($clip.Name)`tExisting file invalid: $issue"
            throw "Existing file is invalid and was preserved: $wavPath ($issue). Inspect and move/rename it before retrying."
        }
        Write-Host "Verified existing clip: $($clip.Name)"
        $skipped++
        continue
    }

    $downloadArgs = @(
        '-m', 'yt_dlp', '--ignore-config', '--no-playlist', '--no-overwrites', '--no-progress',
        '--downloader-args', 'ffmpeg:-loglevel error',
        '--ffmpeg-location', $ffmpegDirectory, '--js-runtimes', $runtime,
        '-f', 'bestaudio', '-x', '--audio-format', 'wav',
        '--download-sections', $clip.Section, '--force-keyframes-at-cuts',
        '--retries', '2', '--fragment-retries', '2', '--socket-timeout', '30',
        '-o', (Join-Path $audioPath "$($clip.Name).%(ext)s"),
        "https://www.youtube.com/watch?v=$($clip.Id)"
    )
    # The helper streams stderr safely on Windows PowerShell and classifies errors.
    $ytArguments = @($downloadArgs | Select-Object -Skip 2)
    & $python $downloadRunner @ytArguments
    $downloadExit = $LASTEXITCODE
    if ($downloadExit -eq 20) {
        $entry = [ordered]@{ sample_id = $clip.Name; video_id = $clip.Id; reason = 'yt-dlp reported private, removed or unavailable'; checked_at = (Get-Date -Format o) }
        Add-Content -LiteralPath $unavailableLog -Value ($entry | ConvertTo-Json -Compress) -Encoding UTF8
        [void]$unavailableIds.Add($clip.Name)
        $unavailable++
        Write-Warning "$($clip.Name): unavailable; recorded and skipped."
        Start-Sleep -Seconds 2
        continue
    }
    $issue = $null
    if ($downloadExit -eq 21) { $issue = 'Access restriction, bot check or rate limit; clip skipped' }
    elseif ($downloadExit -ne 0) { $issue = "yt-dlp exit code $downloadExit" }
    elseif (-not (Test-Path -LiteralPath $wavPath -PathType Leaf)) { $issue = 'No WAV output was produced' }
    else { $issue = Get-ClipValidation $wavPath $clip.Duration }
    if ($issue) {
        Add-Content -LiteralPath $failureLog -Value "$(Get-Date -Format o)`t$($clip.Name)`t$issue"
        Write-Warning "$($clip.Name): $issue"
        $failures++
    }
    else {
        $downloaded++
        if ($archivedIds.Add($clip.Id)) {
            Add-Content -LiteralPath $archivePath -Value "youtube $($clip.Id)" -Encoding ASCII
        }
        Write-Host "Verified downloaded clip: $($clip.Name)"
    }
    Start-Sleep -Seconds 2
}
Write-Host "Finished: $downloaded downloaded, $skipped existing clips skipped ($fastSkipped using archive), $unavailable unavailable clips skipped, $failures other failures."
if ($failures -gt 0) { Write-Warning "Batch completed with failed clips. Review $failureLog before preprocessing." }
