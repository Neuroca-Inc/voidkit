Param(
    [switch]$BinaryScan,
    [int]$MaxBytes = 5242880  # 5 MB cap for binary scan
)

$ErrorActionPreference = "Stop"

Write-Host "Repo-wide FUM audit" -ForegroundColor Cyan
Write-Host ("Mode: " + ($(if ($BinaryScan) { "TEXT + BINARY" } else { "TEXT-ONLY" }))) -ForegroundColor Yellow

# Exclusions
$excludeDirs = @(
    '(^|\\)ignore-temp($|\\)',
    '\.git($|\\)',
    '(^|\\)venv($|\\)',
    '(^|\\)\.venv($|\\)',
    '(^|\\)dist($|\\)',
    '(^|\\)build($|\\)',
    '(^|\\)__pycache__($|\\)',
    '(^|\\)\.pytest_cache($|\\)',
    '(^|\\)\.mypy_cache($|\\)',
    '(^|\\)\.ruff_cache($|\\)',
    '(^|\\)\.idea($|\\)',
    '(^|\\)\.vscode($|\\)'
)

function Is-ExcludedPath {
    param([string]$Path)
    foreach ($rx in $excludeDirs) {
        if ($Path -match $rx) { return $true }
    }
    return $false
}

# Collect all files
$allFiles = Get-ChildItem -Path . -Recurse -File | Where-Object { -not (Is-ExcludedPath $_.FullName) }

# 1) Filename hits containing 'fum' (case-insensitive)
Write-Host "`nScanning filenames for 'fum'..." -ForegroundColor Cyan
$filenameHits = $allFiles | Where-Object { $_.Name -match '(?i)fum' }
if ($filenameHits) {
    $filenameHits | Select-Object FullName
} else {
    Write-Host "No filename hits." -ForegroundColor DarkGray
}

# 2) Text content scan for word-boundary fum/FUM in common text files
Write-Host "`nScanning text files for '\bfum\b' occurrences..." -ForegroundColor Cyan
$textExt = @('*.py','*.md','*.rst','*.txt','*.toml','*.ini','*.cfg','*.yml','*.yaml','*.json','*.csv','*.ipynb')
$textFiles = $allFiles | Where-Object {
    $name = $_.Name.ToLower()
    foreach ($e in $textExt) {
        if ($name -like $e.ToLower()) { return $true }
    }
    return $false
}

$textHits = @()
if ($textFiles) {
    $textHits = Select-String -Path ($textFiles | Select-Object -ExpandProperty FullName) -Pattern '(?i)\bfum\b' -AllMatches
    if ($textHits) {
        $textHits | Select-Object Path, LineNumber, Line
    } else {
        Write-Host "No text content hits." -ForegroundColor DarkGray
    }
} else {
    Write-Host "No text files to scan." -ForegroundColor DarkGray
}

# 3) Optional binary scan for ASCII 'FUM'/'fum' in non-text files up to MaxBytes
$binaryHits = @()
if ($BinaryScan) {
    Write-Host "`nBinary scan (ASCII) for 'FUM/fum' up to $MaxBytes bytes per file..." -ForegroundColor Cyan
    $textSet = $textFiles | Select-Object -ExpandProperty FullName
    $nonText = $allFiles | Where-Object { $textSet -notcontains $_.FullName }
    foreach ($f in $nonText) {
        try {
            if ($f.Length -gt $MaxBytes) { continue }
            $bytes = [System.IO.File]::ReadAllBytes($f.FullName)
            if (-not $bytes) { continue }
            $str = [System.Text.Encoding]::ASCII.GetString($bytes)
            if ($str -match '(?i)fum') {
                $binaryHits += [pscustomobject]@{
                    Path = $f.FullName
                    Size = $f.Length
                    Note = "ASCII match"
                }
            }
        } catch {
            # ignore unreadable files
        }
    }
    if ($binaryHits.Count -gt 0) {
        $binaryHits | Select-Object Path, Size, Note
    } else {
        Write-Host "No binary content hits (within size cap)." -ForegroundColor DarkGray
    }
}

# 4) Summary and optional report file
$report = @()
$report += "# FUM Audit Report"
$report += ("Timestamp: {0:yyyy-MM-dd HH:mm:ss}" -f (Get-Date))
$report += ""
$report += "## Filename hits"
if ($filenameHits) { $report += ($filenameHits | Select-Object -ExpandProperty FullName) } else { $report += "(none)" }
$report += ""
$report += "## Text content hits"
if ($textHits) {
    foreach ($h in $textHits) { $report += ("{0}:{1}: {2}" -f $h.Path, $h.LineNumber, $h.Line) }
} else { $report += "(none)" }
$report += ""
$report += "## Binary content hits"
if ($BinaryScan -and $binaryHits) {
    foreach ($b in $binaryHits) { $report += ("{0} ({1} bytes)" -f $b.Path, $b.Size) }
} elseif ($BinaryScan) { $report += "(none)" } else { $report += "(skipped)" }

$outPath = Join-Path (Resolve-Path .) "fum_audit_report.txt"
Set-Content -LiteralPath $outPath -Value ($report -join [Environment]::NewLine) -Encoding UTF8

Write-Host "`nAudit complete. Report saved to $outPath" -ForegroundColor Green