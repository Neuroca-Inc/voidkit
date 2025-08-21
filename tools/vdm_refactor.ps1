Param(
    [switch]$Apply,
    [switch]$DeleteOld
)

$ErrorActionPreference = "Stop"

Write-Host "VDM refactor utility" -ForegroundColor Cyan
Write-Host ("Mode: " + ($(if ($Apply) { "APPLY" } else { "AUDIT" }))) -ForegroundColor Yellow
if ($Apply -and $DeleteOld) {
    Write-Host "DeleteOld: ENABLED (fum_advanced_math will be removed after copy)" -ForegroundColor Red
}

# Exclusions (do not scan/modify these directories)
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

# Helper: safe content replace using an ordered rule list
function Invoke-ContentReplace {
    param (
        [string]$Path,
        [array]$Rules
    )
    $orig = Get-Content -Raw -LiteralPath $Path
    $new = $orig
    foreach ($rule in $Rules) {
        $pattern = $rule.Pattern
        $replacement = $rule.Replacement
        $new = [regex]::Replace($new, $pattern, $replacement)
    }
    if ($new -ne $orig) {
        Set-Content -LiteralPath $Path -Value $new -Encoding UTF8
        return $true
    }
    return $false
}

# Gather eligible files (respect exclusions)
$allFiles = Get-ChildItem -Path . -Recurse -File | Where-Object { -not (Is-ExcludedPath $_.FullName) }

# 1) Report file names containing 'fum'
Write-Host "`nScanning for filenames containing 'fum' (case-insensitive)..." -ForegroundColor Cyan
$fumNameHits = $allFiles | Where-Object { $_.Name -match '(?i)fum' }
if ($fumNameHits) {
    $fumNameHits | Select-Object FullName
} else {
    Write-Host "No filenames contain 'fum'." -ForegroundColor DarkGray
}

# 2) Report in-file occurrences of 'fum' tokens (text files only)
Write-Host "`nScanning for in-file occurrences of '\bfum\b' (case-insensitive)..." -ForegroundColor Cyan
$textExt = @('py','md','toml','rst','txt','yml','yaml','ini','cfg','json','csv','ipynb')
$initialScanFiles = $allFiles | Where-Object {
    $ext = $_.Extension.TrimStart('.').ToLower()
    $textExt -contains $ext
}
if ($initialScanFiles) {
    $hits = Select-String -Path ($initialScanFiles | Select-Object -ExpandProperty FullName) -Pattern '(?i)\bfum\b' -List
    if ($hits) {
        $hits | Select-Object Path, LineNumber, Line
    } else {
        Write-Host "No in-file 'fum' tokens found." -ForegroundColor DarkGray
    }
} else {
    Write-Host "No files matched for content scan." -ForegroundColor DarkGray
}

# Suggested renames (display only)
Write-Host "`nSuggested renames if present:" -ForegroundColor Cyan
if (Test-Path "voidkit\void_dynamics\FUM_Void_Equations.py") {
    Write-Host " - voidkit\void_dynamics\FUM_Void_Equations.py -> voidkit\void_dynamics\VDM_Void_Equations.py"
}
if (Test-Path "fum_advanced_math") {
    Write-Host " - Move fum_advanced_math -> voidkit\advanced_math (then update imports)"
}

if (-not $Apply) {
    Write-Host "`nAudit complete. Re-run with -Apply to perform changes." -ForegroundColor Yellow
    exit 0
}

# APPLY MODE
Write-Host "`nApplying changes..." -ForegroundColor Yellow

# 3) Targeted renames
if (Test-Path "voidkit\void_dynamics\FUM_Void_Equations.py") {
    Write-Host "Renaming FUM_Void_Equations.py -> VDM_Void_Equations.py"
    if (-not (Test-Path "voidkit\void_dynamics\VDM_Void_Equations.py")) {
        Rename-Item -Path "voidkit\void_dynamics\FUM_Void_Equations.py" -NewName "VDM_Void_Equations.py" -Force
    } else {
        Write-Host "Target already exists: VDM_Void_Equations.py (skipping rename)" -ForegroundColor DarkYellow
    }
}

if (Test-Path "fum_advanced_math") {
    if (-not (Test-Path "voidkit\advanced_math")) {
        Write-Host "Creating voidkit\advanced_math"
        New-Item -ItemType Directory -Path "voidkit\advanced_math" | Out-Null
    }
    Write-Host "Copying fum_advanced_math -> voidkit\advanced_math"
    Copy-Item -Path "fum_advanced_math\*" -Destination "voidkit\advanced_math\" -Recurse -Force

    if ($DeleteOld) {
        Write-Host "Deleting fum_advanced_math (per -DeleteOld)" -ForegroundColor Red
        Remove-Item -Recurse -Force "fum_advanced_math"
    } else {
        Write-Host "Note: fum_advanced_math left in place (run with -DeleteOld to remove after verifying)." -ForegroundColor DarkYellow
    }
}

# 4) Content replacements (code and docs)
# Ordered rules: specific before general
$replaceRules = @(
    @{ Pattern = 'fum_advanced_math';            Replacement = 'voidkit.advanced_math' },
    @{ Pattern = '\bFUM_Void_Equations\b';       Replacement = 'VDM_Void_Equations'    },
    @{ Pattern = '\bFUM\b';                      Replacement = 'VDM'                    },
    @{ Pattern = '\bfum\b';                      Replacement = 'vdm'                    }
)

# Recompute file list post-renames, respecting exclusions and text extensions
$scanFiles = Get-ChildItem -Path . -Recurse -File | Where-Object {
    -not (Is-ExcludedPath $_.FullName) -and ($textExt -contains $_.Extension.TrimStart('.').ToLower())
}

$modified = 0
foreach ($f in $scanFiles) {
    $changed = Invoke-ContentReplace -Path $f.FullName -Rules $replaceRules
    if ($changed) {
        $modified++
        Write-Host ("Updated: " + $f.FullName)
    }
}
Write-Host ("Files updated: {0}" -f $modified) -ForegroundColor Green

# 5) Post-apply residual scan
Write-Host "`nResidual scan for '\bfum\b' tokens after apply:" -ForegroundColor Cyan
$remaining = Select-String -Path ($scanFiles | Select-Object -ExpandProperty FullName) -Pattern '(?i)\bfum\b' -List
if ($remaining) {
    $remaining | Select-Object Path, LineNumber, Line
    Write-Host "`nManual review recommended for the above lines." -ForegroundColor DarkYellow
} else {
    Write-Host "No remaining 'fum' tokens detected." -ForegroundColor Green
}

Write-Host "`nVDM refactor apply complete." -ForegroundColor Green