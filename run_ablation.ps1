# Stage-1-only ablation runs (see ablation_stage1.py). Same model and settings as the
# no-issue run; only the input dataset differs. Resumable: finished bugs are skipped.
#
# Usage:
#   conda activate agentless
#   cd "C:\Users\USER\Documents\IIT\4thYear\FYP\Agentless"
#   python ablation_stage1.py make
#   .\run_ablation.ps1
#   python ablation_stage1.py compare

$env:OPENAI_API_KEY = "dummy"          # Ollama ignores it, the client library requires it
$started = Get-Date

foreach ($variant in @("control", "no_fixframes", "no_testpaths")) {
    $dataset = "datasets/ablation_$variant"
    $bugs = Get-Content "$dataset/test.jsonl" | ForEach-Object { ($_ | ConvertFrom-Json).instance_id }
    foreach ($bug in $bugs) {
        $out = "results/ablation_$variant/$bug"
        if (Test-Path "$out/loc_outputs.jsonl") {
            Write-Host "[skip] $variant $bug"
            continue
        }
        Write-Host "[run ] $variant $bug" -ForegroundColor Yellow
        python -m agentless.fl.localize --file_level `
            --output_folder $out `
            --top_n 3 `
            --model "qwen2.5-coder:7b" --backend ollama `
            --dataset $dataset `
            --target_id $bug --num_threads 1
    }
}

Write-Host "=== Done in $([int]((Get-Date) - $started).TotalMinutes) min ===" -ForegroundColor Cyan
