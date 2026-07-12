param(
    [string]$FullHeldoutRoot = ".\eval_runs\full_heldout",
    [string]$AnalysisRoot = ".\eval_runs\full_heldout\analysis",
    [string]$Stage1Zip = "",
    [string]$Stage2Zip = "",
    [string]$QwenZip = "",
    [string]$AstroLlavaZip = "",
    [string]$RecordsJson = ".\datasets\astrollava_llava\test.json",
    [int]$BootstrapSamples = 10000,
    [int]$QualitativePerCategory = 12,
    [int]$JudgeSampleSize = 150,
    [string]$Python = "python",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepoRoot

try {
    if (-not $Stage1Zip) {
        $Stage1Zip = Join-Path $FullHeldoutRoot "astraq-vl-stage1-full-heldout-eval-v1.zip"
    }
    if (-not $Stage2Zip) {
        $Stage2Zip = Join-Path $FullHeldoutRoot "astraq-vl-stage2-full-heldout-eval-v1.zip"
    }
    if (-not $QwenZip) {
        $QwenZip = Join-Path $FullHeldoutRoot "qwen2_5-vl-7b-full-heldout-eval-v1.zip"
    }
    if (-not $AstroLlavaZip) {
        $AstroLlavaZip = Join-Path $FullHeldoutRoot "astrollava-reference-full-heldout-eval-v1.zip"
    }

    Write-Host "== Preflight =="
    Write-Host "ZIP artifacts found:"
    if (Test-Path $FullHeldoutRoot) {
        Get-ChildItem $FullHeldoutRoot -Recurse -Filter "*.zip" |
            Select-Object FullName, Length |
            Format-Table -AutoSize
        Write-Host "Per-sample metric files found:"
        Get-ChildItem $FullHeldoutRoot -Recurse -Filter "*.per_sample.jsonl" |
            Select-Object FullName, Length |
            Format-Table -AutoSize
    } else {
        Write-Host "  $FullHeldoutRoot does not exist."
    }

    & $Python -m py_compile `
        ".\scripts\bootstrap_full_heldout_ci.py" `
        ".\scripts\inspect_skipped_rows.py" `
        ".\scripts\mine_qualitative_examples.py" `
        ".\scripts\sample_judge_set.py"

    $required = @(
        @{ Name = "Stage-1"; Path = $Stage1Zip },
        @{ Name = "Stage-2"; Path = $Stage2Zip },
        @{ Name = "Qwen2.5-VL"; Path = $QwenZip },
        @{ Name = "AstroLLaVA reference"; Path = $AstroLlavaZip }
    )
    $missing = @($required | Where-Object { -not (Test-Path $_.Path) })
    if ($missing.Count -gt 0) {
        Write-Host ""
        Write-Host "Missing required artifacts:"
        foreach ($item in $missing) {
            Write-Host "  - $($item.Name): $($item.Path)"
        }
        throw "Place the four full-heldout ZIPs under $FullHeldoutRoot, or pass explicit -Stage1Zip/-Stage2Zip/-QwenZip/-AstroLlavaZip paths."
    }

    if ($DryRun) {
        Write-Host "Dry run passed. Required artifacts exist; no analyses were run."
        return
    }

    New-Item -ItemType Directory -Force -Path $AnalysisRoot | Out-Null

    Write-Host ""
    Write-Host "== Bootstrap CIs =="
    & $Python ".\scripts\bootstrap_full_heldout_ci.py" `
        --stage1-zip $Stage1Zip `
        --stage2-zip $Stage2Zip `
        --qwen-zip $QwenZip `
        --astrollava-zip $AstroLlavaZip `
        --n-bootstrap $BootstrapSamples `
        --out (Join-Path $AnalysisRoot "bootstrap_ci")

    Write-Host ""
    Write-Host "== AstroLLaVA skipped rows =="
    $skippedArgs = @(
        ".\scripts\inspect_skipped_rows.py",
        "--artifact", $AstroLlavaZip,
        "--out", (Join-Path $AnalysisRoot "astrollava_reference_skipped_rows")
    )
    if (Test-Path $RecordsJson) {
        $skippedArgs += @("--records-json", $RecordsJson)
    } else {
        Write-Host "Records JSON not found at $RecordsJson; running skipped-row inspection without reference matching."
    }
    & $Python @skippedArgs

    Write-Host ""
    Write-Host "== Qualitative examples =="
    & $Python ".\scripts\mine_qualitative_examples.py" `
        --stage2 $Stage2Zip `
        --stage1 $Stage1Zip `
        --qwen $QwenZip `
        --astrollava $AstroLlavaZip `
        --per-category $QualitativePerCategory `
        --out (Join-Path $AnalysisRoot "qualitative_examples")

    Write-Host ""
    Write-Host "== Judge sample =="
    & $Python ".\scripts\sample_judge_set.py" `
        --artifact $Stage1Zip `
        --artifact $Stage2Zip `
        --artifact $QwenZip `
        --artifact $AstroLlavaZip `
        --sample-size $JudgeSampleSize `
        --out (Join-Path $AnalysisRoot "judge_sample")

    Write-Host ""
    Write-Host "Offline analysis complete. Main outputs:"
    Write-Host "  - $(Join-Path $AnalysisRoot "bootstrap_ci.md")"
    Write-Host "  - $(Join-Path $AnalysisRoot "astrollava_reference_skipped_rows.md")"
    Write-Host "  - $(Join-Path $AnalysisRoot "qualitative_examples.md")"
    Write-Host "  - $(Join-Path $AnalysisRoot "judge_sample.csv")"
    Write-Host "  - $(Join-Path $AnalysisRoot "judge_sample.rubric.md")"
} finally {
    Pop-Location
}
