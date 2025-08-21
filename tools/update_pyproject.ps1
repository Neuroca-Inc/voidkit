Param(
    [switch]$Apply
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$pyprojectPath = Join-Path $root "pyproject.toml"
$reqPath = Join-Path $root "requirements.txt"
$licensePath = Join-Path $root "LICENSE"
$gitConfigPath = Join-Path $root ".git\config"

Write-Host "Updating pyproject.toml from repository sources" -ForegroundColor Cyan
Write-Host "Mode: " -NoNewline; Write-Host ($(if ($Apply) { "APPLY" } else { "AUDIT" })) -ForegroundColor Yellow

# ---------- Helpers ----------

function Get-FileText {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    return Get-Content -Raw -LiteralPath $Path -ErrorAction Stop
}

function Parse-Requirements {
    param([string]$Text)
    if (-not $Text) { return @() }
    $devMarkers = @(
        "pytest","pytest-cov","black","ruff","mypy","tox","pre-commit","coverage","hypothesis","tomlkit","pip-tools"
    )
    $deps = New-Object System.Collections.Generic.List[string]
    foreach ($line in ($Text -split "`r?`n")) {
        $s = $line.Trim()
        if (-not $s) { continue }
        if ($s.StartsWith("#")) { continue }
        if ($s -match '^\s*(-r|--requirement|-e|--editable|-c|--constraint|--)') { continue }
        if ($s -match '^\s*-[A-Za-z]') { continue }
        # strip inline comments
        if ($s -match '\s+#') { $s = ($s -split '\s+#',2)[0].Trim() }
        if (-not $s) { continue }
        $name = ($s -split '[<>=!~\[\];\s]',2)[0].ToLower()
        if ($devMarkers -contains $name) { continue }
        $deps.Add($s)
    }
    return $deps.ToArray()
}

function Detect-LicenseClassifier {
    param([string]$Text)
    if (-not $Text) { return $null }
    $t = $Text.ToLower()
    function Has { param([string[]]$terms); return ($terms | ForEach-Object { $t.Contains($_) }) -notcontains $false }
    if (Has @("mit license","permission is hereby granted")) { return 'License :: OSI Approved :: MIT License' }
    if (Has @("apache license","version 2.0")) { return 'License :: OSI Approved :: Apache Software License' }
    if ($t -match 'redistribution and use in source and binary forms') { return 'License :: OSI Approved :: BSD License' }
    if ($t -match 'gnu general public license' -and $t -match 'version 3') { return 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)' }
    if ($t -match 'gnu general public license' -and $t -match 'version 2') { return 'License :: OSI Approved :: GNU General Public License v2 (GPLv2)' }
    if ($t -match 'gnu lesser general public license' -and $t -match 'version 3') { return 'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)' }
    if ($t -match 'gnu lesser general public license' -and $t -match 'version 2.1') { return 'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)' }
    if ($t -match 'mozilla public license' -and $t -match '2.0') { return 'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)' }
    if ($t -match 'this is free and unencumbered software released into the public domain' -or $t -match 'the unlicense') { return 'License :: OSI Approved :: The Unlicense (Unlicense)' }
    if ($t -match 'creative commons' -and $t -match 'cc0') { return 'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication' }
    return $null
}

function Detect-Authors {
    param([string]$Text)
    $authors = @()
    if (-not $Text) { return $authors }
    foreach ($m in [regex]::Matches($Text, '^[ \t]*Author[s]?:[ \t]*(.+)$', 'IgnoreCase, Multiline')) {
        $val = $m.Groups[1].Value.Trim()
        $authors += ,$val
    }
    if (-not $authors) {
        foreach ($m in [regex]::Matches($Text, 'copyright\s*\(c\)\s*\d{2,4}[^A-Za-z0-9]*([^\r\n<]+?)(?:\s*<([^>]+)>)?\s*$', 'IgnoreCase, Multiline')) {
            $name = ($m.Groups[1].Value).Trim().Trim(',')
            $mail = ($m.Groups[2].Value).Trim()
            if ($name) {
                if ($mail) { $authors += ,("$name <$mail>") } else { $authors += ,$name }
            }
        }
    }
    # dedupe
    $seen = @{}
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($a in $authors) {
        if (-not $seen.ContainsKey($a)) { $seen[$a]=$true; $out.Add($a) }
    }
    return $out.ToArray()
}

function Normalize-GitUrl {
    param([string]$u)
    if (-not $u) { return $u }
    $url = $u.Trim()
    if ($url.ToLower().EndsWith(".git")) { $url = $url.Substring(0, $url.Length-4) }
    $m = [regex]::Match($url, '^git@([^:]+):(.+)$')
    if ($m.Success) {
        $gitHost = $m.Groups[1].Value
        $repoPath = $m.Groups[2].Value
        return ("https://{0}/{1}" -f $gitHost, $repoPath).TrimEnd('/')
    }
    return $url.TrimEnd('/')
}

function Detect-Urls {
    param([string]$GitConfig)
    if (-not $GitConfig) { return @{} }
    $m = [regex]::Match($GitConfig, '\[remote\s+"origin"\][^\[]*?^\s*url\s*=\s*(.+)$', 'Multiline')
    if (-not $m.Success) { return @{} }
    $raw = $m.Groups[1].Value.Trim()
    $clean = Normalize-GitUrl $raw
    $urls = @{
        "Homepage" = $clean
        "Repository" = $clean
    }
    if ($clean.ToLower().Contains("github.com")) {
        $urls["Issues"] = ($clean.TrimEnd('/') + "/issues")
    } elseif ($clean.ToLower().Contains("gitlab.com")) {
        $urls["Issues"] = ($clean.TrimEnd('/') + "/-/issues")
    }
    return $urls
}

function Indent($level) { return (" " * ($level)) }

# Remove a [[project.*]] or [project.*] section entirely
function Remove-Section {
    param(
        [string[]]$Lines,
        [string]$HeaderRegex  # e.g., '^\[project\.urls\]\s*$'
    )
    $start = $null
    for ($i=0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -match $HeaderRegex) { $start = $i; break }
    }
    if ($null -eq $start) { return ,$Lines }
    # find end = next section header
    $end = $Lines.Count
    for ($j=$start+1; $j -lt $Lines.Count; $j++) {
        if ($Lines[$j] -match '^\[') { $end = $j; break }
    }
    $new = @()
    if ($start -gt 0) { $new += $Lines[0..($start-1)] }
    if ($end -lt $Lines.Count) { $new += $Lines[$end..($Lines.Count-1)] }
    return ,$new
}

# Remove a key=value block (array) inside [project] by key name
function Remove-Project-KeyOrArray {
    param(
        [string[]]$Lines,
        [string]$Key  # e.g., 'dependencies' or 'authors'
    )
    # find project bounds
    $projStart = $null
    for ($i=0; $i -lt $Lines.Count; $i++) { if ($Lines[$i] -match '^\[project\]\s*$') { $projStart=$i; break } }
    if ($null -eq $projStart) { return ,$Lines }
    $projEnd = $Lines.Count
    for ($j=$projStart+1; $j -lt $Lines.Count; $j++) { if ($Lines[$j] -match '^\[') { $projEnd=$j; break } }

    $new = @()
    $i = 0
    while ($i -lt $Lines.Count) {
        if ($i -ge $projStart -and $i -lt $projEnd) {
            if ($Lines[$i] -match "^\s*$Key\s*=") {
                # determine if array begins here: consume until bracket balance closes
                $line = $Lines[$i]
                $isArray = ($line -match '\[') -or ($i+1 -lt $Lines.Count -and $Lines[$i+1] -match '^\s*\[')
                if ($isArray) {
                    $balance = 0
                    # count brackets from current line onward
                    while ($i -lt $Lines.Count) {
                        $balance += ([regex]::Matches($Lines[$i], '\[')).Count
                        $balance -= ([regex]::Matches($Lines[$i], '\]')).Count
                        $i++
                        if ($balance -le 0) { break }
                    }
                    continue
                } else {
                    # single-line key=value
                    $i++
                    continue
                }
            }
        }
        $new += $Lines[$i]
        $i++
    }
    return ,$new
}

# Insert (or replace) dependencies array in [project]
function Upsert-Project-Dependencies {
    param(
        [string[]]$Lines,
        [string[]]$Deps
    )
    $Lines = Remove-Project-KeyOrArray -Lines $Lines -Key 'dependencies'
    # Find insertion point: after 'license =' in [project], else after 'readme =', else at end of [project]
    $projStart = $null
    for ($i=0; $i -lt $Lines.Count; $i++) { if ($Lines[$i] -match '^\[project\]\s*$') { $projStart=$i; break } }
    if ($null -eq $projStart) { return ,$Lines }

    $projEnd = $Lines.Count
    for ($j=$projStart+1; $j -lt $Lines.Count; $j++) { if ($Lines[$j] -match '^\[') { $projEnd=$j; break } }

    $insertAt = $projStart+1
    for ($k=$projStart+1; $k -lt $projEnd; $k++) {
        if ($Lines[$k] -match '^\s*license\s*=') { $insertAt = $k+1 }
        elseif ($Lines[$k] -match '^\s*readme\s*=') { if ($insertAt -lt ($k+1)) { $insertAt = $k+1 } }
    }

    $block = @()
    $block += "dependencies = ["
    foreach ($d in $Deps) { $block += ("  `"$d`",") }
    if ($Deps.Count -gt 0) {
        # replace trailing comma on last line
        $block[$block.Count-1] = $block[$block.Count-1].TrimEnd(',')
    }
    $block += "]"

    $new = @()
    if ($insertAt -gt 0) { $new += $Lines[0..($insertAt-1)] }
    $new += $block
    if ($insertAt -lt $Lines.Count) { $new += $Lines[$insertAt..($Lines.Count-1)] }
    return ,$new
}

# Insert (or replace) authors array in [project]
function Upsert-Project-Authors {
    param(
        [string[]]$Lines,
        [string[]]$AuthorStrings  # e.g., "Jane Doe <jane@x.com>"
    )
    $Lines = Remove-Project-KeyOrArray -Lines $Lines -Key 'authors'

    $projStart = $null
    for ($i=0; $i -lt $Lines.Count; $i++) { if ($Lines[$i] -match '^\[project\]\s*$') { $projStart=$i; break } }
    if ($null -eq $projStart) { return ,$Lines }
    $projEnd = $Lines.Count
    for ($j=$projStart+1; $j -lt $Lines.Count; $j++) { if ($Lines[$j] -match '^\[') { $projEnd=$j; break } }

    $insertAt = $projStart+1
    for ($k=$projStart+1; $k -lt $projEnd; $k++) {
        if ($Lines[$k] -match '^\s*license\s*=') { $insertAt = $k+1 }
        elseif ($Lines[$k] -match '^\s*dependencies\s*=') { if ($insertAt -lt ($k+1)) { $insertAt = $k+1 } }
    }

    $block = @()
    $block += "authors = ["
    foreach ($s in $AuthorStrings) {
        $name = $s
        $email = $null
        $m = [regex]::Match($s, '(.+?)\s*<([^>]+)>')
        if ($m.Success) { $name = $m.Groups[1].Value.Trim(); $email = $m.Groups[2].Value.Trim() }
        if ($email) {
            $block += ("  { name = `"$name`", email = `"$email`" },")
        } else {
            $block += ("  { name = `"$name`" },")
        }
    }
    if ($AuthorStrings.Count -gt 0) {
        $block[$block.Count-1] = $block[$block.Count-1].TrimEnd(',')
    }
    $block += "]"

    $new = @()
    if ($insertAt -gt 0) { $new += $Lines[0..($insertAt-1)] }
    $new += $block
    if ($insertAt -lt $Lines.Count) { $new += $Lines[$insertAt..($Lines.Count-1)] }
    return ,$new
}

# Upsert [project.urls] table (replace if exists)
function Upsert-Project-Urls {
    param(
        [string[]]$Lines,
        [hashtable]$Urls
    )
    $Lines = Remove-Section -Lines $Lines -HeaderRegex '^\[project\.urls\]\s*$'

    # find insertion after [project] block
    $projStart = $null
    for ($i=0; $i -lt $Lines.Count; $i++) { if ($Lines[$i] -match '^\[project\]\s*$') { $projStart=$i; break } }
    if ($null -eq $projStart) { return ,$Lines }
    $insertAt = $Lines.Count
    for ($j=$projStart+1; $j -lt $Lines.Count; $j++) { if ($Lines[$j] -match '^\[') { $insertAt=$j; break } }

    $block = @("[project.urls]")
    foreach ($k in $Urls.Keys) {
        $block += ("$k = `"$($Urls[$k])`"")
    }

    $new = @()
    if ($insertAt -gt 0) { $new += $Lines[0..($insertAt-1)] }
    $new += $block
    if ($insertAt -lt $Lines.Count) { $new += $Lines[$insertAt..($Lines.Count-1)] }
    return ,$new
}

# Remove dynamic field (entire line)
function Remove-Project-Dynamic {
    param([string[]]$Lines)
    $projStart = $null
    for ($i=0; $i -lt $Lines.Count; $i++) { if ($Lines[$i] -match '^\[project\]\s*$') { $projStart=$i; break } }
    if ($null -eq $projStart) { return ,$Lines }
    $projEnd = $Lines.Count
    for ($j=$projStart+1; $j -lt $Lines.Count; $j++) { if ($Lines[$j] -match '^\[') { $projEnd=$j; break } }
    $new = @()
    for ($i=0; $i -lt $Lines.Count; $i++) {
        if ($i -ge $projStart -and $i -lt $projEnd) {
            if ($Lines[$i] -match '^\s*dynamic\s*=') { continue }
        }
        $new += $Lines[$i]
    }
    return ,$new
}

# Append license classifier to project.classifiers if missing
function Upsert-License-Classifier {
    param(
        [string[]]$Lines,
        [string]$Classifier
    )
    if (-not $Classifier) { return ,$Lines }
    $projStart = $null
    for ($i=0; $i -lt $Lines.Count; $i++) { if ($Lines[$i] -match '^\[project\]\s*$') { $projStart=$i; break } }
    if ($null -eq $projStart) { return ,$Lines }
    $projEnd = $Lines.Count
    for ($j=$projStart+1; $j -lt $Lines.Count; $j++) { if ($Lines[$j] -match '^\[') { $projEnd=$j; break } }

    # find classifiers array
    $start = $null
    for ($k=$projStart+1; $k -lt $projEnd; $k++) {
        if ($Lines[$k] -match '^\s*classifiers\s*=\s*\[') { $start=$k; break }
    }
    if ($null -eq $start) {
        # create new block near end of project
        $ins = $projEnd
        $block = @("classifiers = [","  `"$Classifier`"","]")
        $new = @()
        if ($ins -gt 0) { $new += $Lines[0..($ins-1)] }
        $new += $block
        if ($ins -lt $Lines.Count) { $new += $Lines[$ins..($Lines.Count-1)] }
        return ,$new
    }
    # collect until closing ]
    $end = $start
    while ($end -lt $projEnd) {
        if ($Lines[$end] -match '\]') { break }
        $end++
    }
    $existing = ($Lines[$start..$end] -join "`n")
    if ($existing -match [regex]::Escape($Classifier)) { return ,$Lines }

    # insert before closing bracket
    $Lines[$end] = "  `"$Classifier`"`n]"
    return ,$Lines
}

# Remove metadata hook section
function Remove-MetadataHookSection {
    param([string[]]$Lines)
    return Remove-Section -Lines $Lines -HeaderRegex '^\[project\.entry-points\."hatch\.metadata\.hooks"\]\s*$'
}

# ---------- Load sources ----------

$pyText = Get-FileText -Path $pyprojectPath
if (-not $pyText) { throw "pyproject.toml not found at $pyprojectPath" }
$reqText = Get-FileText -Path $reqPath
$licenseText = Get-FileText -Path $licensePath
$gitConfigText = Get-FileText -Path $gitConfigPath

$deps = Parse-Requirements -Text $reqText
$authors = Detect-Authors -Text $licenseText
$urls = Detect-Urls -GitConfig $gitConfigText
$licenseClassifier = Detect-LicenseClassifier -Text $licenseText

Write-Host ("Found dependencies: " + ($deps -join ", ")) -ForegroundColor DarkCyan
Write-Host ("Found authors: " + ($(if ($authors) { ($authors -join "; ") } else { "(none)" }))) -ForegroundColor DarkCyan
if ($urls.Keys.Count -gt 0) { Write-Host ("Found URLs: " + ($urls.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" } -join ", ")) -ForegroundColor DarkCyan }
if ($licenseClassifier) { Write-Host ("Detected license classifier: $licenseClassifier") -ForegroundColor DarkCyan }

# ---------- Transform ----------

$lines = $pyText -split "`r?`n"

$lines = Remove-Project-Dynamic -Lines $lines
$lines = Remove-MetadataHookSection -Lines $lines
$lines = Upsert-Project-Dependencies -Lines $lines -Deps $deps
$lines = Upsert-Project-Authors -Lines $lines -AuthorStrings $authors
if ($urls.Keys.Count -gt 0) { $lines = Upsert-Project-Urls -Lines $lines -Urls $urls }
if ($licenseClassifier) { $lines = Upsert-License-Classifier -Lines $lines -Classifier $licenseClassifier }

$newText = ($lines -join "`r`n")

if (-not $Apply) {
    Write-Host "`n--- Diff (preview) ---" -ForegroundColor Yellow
    # crude preview
    $origHash = [BitConverter]::ToString((New-Object -TypeName System.Security.Cryptography.SHA1Managed).ComputeHash([Text.Encoding]::UTF8.GetBytes($pyText)))
    $newHash  = [BitConverter]::ToString((New-Object -TypeName System.Security.Cryptography.SHA1Managed).ComputeHash([Text.Encoding]::UTF8.GetBytes($newText)))
    Write-Host "pyproject.toml SHA1 before: $origHash"
    Write-Host "pyproject.toml SHA1 after : $newHash"
    if ($origHash -eq $newHash) { Write-Host "No changes required." -ForegroundColor Green } else { Write-Host "Changes would be applied (run with -Apply)." -ForegroundColor Yellow }
    exit 0
}

Set-Content -LiteralPath $pyprojectPath -Value $newText -Encoding UTF8
Write-Host "`npyproject.toml updated." -ForegroundColor Green

Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host (" - Dependencies: {0}" -f ($(if ($deps) { $deps.Count } else { 0 })))
Write-Host (" - Authors: {0}" -f ($(if ($authors) { $authors.Count } else { 0 })))
Write-Host (" - URLs: {0}" -f $urls.Keys.Count)
Write-Host (" - License classifier: {0}" -f ($(if ($licenseClassifier) { $licenseClassifier } else { "(none detected)" })))