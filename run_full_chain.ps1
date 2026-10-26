# Run the full Agentless chain (stage 2 -> 3 -> merge -> repair) over every bug that
# already has stage-1 file-level localization, using ONE identical config for all of them.
#
# Config is the one that produced the first working patch on pytest-dev__pytest-6202:
#   --fine_grain_separate_file   stage 3 asks about one candidate file per call, so a file
#                                cannot be silently dropped from found_edit_locs
#   --context_window 25          wide enough that a ~15-20 line localization miss can still
#                                show the real buggy line to the repair model
#
# Stage 1 is NOT re-run: it is read from results/pilot_rerun/<bug>/loc_outputs.jsonl.
# Output goes to a SEPARATE folder so the existing pilot_rerun results stay untouched.
#
# Resumable: every stage is skipped if its output already exists, so if this dies partway
# (network, Ollama restart, laptop sleep) just run it again and it picks up where it stopped.
#
# Usage:
#   conda activate agentless
#   cd "C:\Users\USER\Documents\IIT\4thYear\FYP\Agentless"
#   .\run_full_chain.ps1

param(
    # Output folder. Defaults to the v2 run (stage 2 now uses
    # --related_level_separate_file). Pass -OutDir to write somewhere else and keep
    # previous runs intact for comparison.
    [string]$OutDir = "results/pilot_full_v2",

    # Explicit bug list. When given, stage 1 is RUN for each bug (into $OutDir)
    # rather than reused from $SrcDir -- use this for bugs that have never been run.
    # When omitted, bugs are discovered from $SrcDir as before and stage 1 is reused.
    [string[]]$Bugs = @(),

    # Dataset Agentless reads. Default = SWE-bench Verified. For BugsInPy, pass the local
    # folder made by bugsinpy_to_agentless.py, e.g. -Dataset datasets/bugsinpy_pilot
    [string]$Dataset = "princeton-nlp/SWE-bench_Verified"
)

$ErrorActionPreference = "Continue"

$SrcDir   = "results/pilot_rerun"      # where stage-1 output already lives
$Model    = "qwen2.5-coder:7b"
$Backend  = "ollama"
$TopN     = 3
$CtxWin   = 25

$env:OPENAI_API_KEY = "dummy"          # Ollama ignores it, the client library requires it

if ($Bugs.Count -gt 0) {
    $bugs = $Bugs | Sort-Object
    Write-Host "Explicit bug list given - stage 1 will be RUN for these, not reused.`n"
} else {
    # every bug that already has stage-1 output
    $bugs = Get-ChildItem $SrcDir -Directory |
            Where-Object { Test-Path (Join-Path $_.FullName "loc_outputs.jsonl") } |
            Select-Object -ExpandProperty Name |
            Sort-Object
}

Write-Host "=== Running full chain for $($bugs.Count) bugs ===" -ForegroundColor Cyan
Write-Host "Config: --fine_grain_separate_file, --context_window $CtxWin, single greedy sample"
Write-Host "Output: $OutDir`n"

$started = Get-Date

foreach ($bug in $bugs) {
    Write-Host "--------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "BUG: $bug" -ForegroundColor Yellow

    # ---- stage 1: file-level localization ----
    # Reused from $SrcDir for bugs already run; generated into $OutDir for new ones.
    if ($Bugs.Count -gt 0) {
        $stage1 = "$OutDir/$bug/loc_outputs.jsonl"
        if (Test-Path $stage1) {
            Write-Host "  [skip] stage 1 already done"
        } else {
            Write-Host "  [run ] stage 1 - file level"
            python -m agentless.fl.localize --file_level `
                --output_folder "$OutDir/$bug" `
                --top_n $TopN `
                --model $Model --backend $Backend `
                --dataset $Dataset `
                --target_id $bug --num_threads 1
        }
    } else {
        $stage1 = "$SrcDir/$bug/loc_outputs.jsonl"
    }

    $related = "$OutDir/$bug/related"
    $editloc = "$OutDir/$bug/edit_loc"
    $merged  = "$OutDir/$bug/merged"
    $repair  = "$OutDir/$bug/repair"

    # ---- stage 2: related-elements localization ----
    if (Test-Path "$related/loc_outputs.jsonl") {
        Write-Host "  [skip] stage 2 already done"
    } else {
        Write-Host "  [run ] stage 2 - related elements (separate file)"
        # --related_level_separate_file: ask about ONE candidate file per model call.
        # Added after django-13401, where stage 2 returned an EMPTY answer for the file
        # stage 1 had correctly ranked first, populated a different file instead, and
        # everything downstream then worked on the wrong file. Mirrors what
        # --fine_grain_separate_file does at stage 3.
        python -m agentless.fl.localize --related_level --related_level_separate_file `
            --output_folder $related `
            --top_n $TopN --compress_assign --compress `
            --start_file $stage1 `
            --model $Model --backend $Backend `
            --dataset $Dataset `
            --target_id $bug --num_threads 1
    }

    # ---- stage 3: fine-grain line-level localization (one call per candidate file) ----
    if (Test-Path "$editloc/loc_outputs.jsonl") {
        Write-Host "  [skip] stage 3 already done"
    } elseif (Test-Path "$related/loc_outputs.jsonl") {
        Write-Host "  [run ] stage 3 - line level (separate file)"
        python -m agentless.fl.localize --fine_grain_line_level --fine_grain_separate_file `
            --output_folder $editloc `
            --top_n $TopN --compress `
            --start_file "$related/loc_outputs.jsonl" `
            --model $Model --backend $Backend `
            --dataset $Dataset `
            --target_id $bug --num_threads 1
    } else {
        Write-Host "  [SKIP] stage 3 - stage 2 produced no output" -ForegroundColor Red
    }

    # ---- merge (no LLM call) ----
    if (Test-Path "$merged/loc_merged_0-0_outputs.jsonl") {
        Write-Host "  [skip] merge already done"
    } elseif (Test-Path "$editloc/loc_outputs.jsonl") {
        Write-Host "  [run ] merge"
        python -m agentless.fl.localize --merge `
            --output_folder $merged `
            --top_n $TopN --num_samples 1 `
            --start_file "$editloc/loc_outputs.jsonl"
    } else {
        Write-Host "  [SKIP] merge - stage 3 produced no output" -ForegroundColor Red
    }

    # ---- repair ----
    if (Test-Path "$repair/output_0_processed.jsonl") {
        Write-Host "  [skip] repair already done"
    } elseif (Test-Path "$merged/loc_merged_0-0_outputs.jsonl") {
        Write-Host "  [run ] repair"
        python -m agentless.repair.repair `
            --loc_file "$merged/loc_merged_0-0_outputs.jsonl" `
            --output_folder $repair `
            --loc_interval --top_n $TopN --context_window $CtxWin --max_samples 1 `
            --cot --diff_format --gen_and_process `
            --model $Model --backend $Backend `
            --dataset $Dataset `
            --num_threads 1
    } else {
        Write-Host "  [SKIP] repair - no merged locations" -ForegroundColor Red
    }
}

$elapsed = (Get-Date) - $started
Write-Host "`n=== Done in $([int]$elapsed.TotalMinutes) min ===" -ForegroundColor Cyan
Write-Host "Now run:  python score_repair.py" -ForegroundColor Cyan
