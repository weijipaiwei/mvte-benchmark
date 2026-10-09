# Evaluate one model on MVTE-benchmark (Windows equivalent of run_all.sh).
#
#   powershell -File evaluation\run_all.ps1 <model_validation.json> <save_dir> [batch_size]
param(
    [Parameter(Mandatory = $true)][string]$JsonPath,
    [Parameter(Mandatory = $true)][string]$SaveDir,
    [int]$Batch = 32
)

# always run from the repository root so the relative image roots resolve
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== 1/5 OCR (ACC / NED) =="
python evaluation/run_ocr.py --json_path $JsonPath --save_dir $SaveDir --ocr_name deepseek-ocr
if ($LASTEXITCODE -ne 0) { throw "OCR step failed" }

Write-Host "== 2/5 SigLIP2 =="
python evaluation/run_siglip2.py --json_path $JsonPath --save_dir $SaveDir --model_path weights/google__siglip2-so400m-patch16-naflex
if ($LASTEXITCODE -ne 0) { throw "SigLIP2 step failed" }

Write-Host "== 3/5 HPSv3 (needs the transformers==4.45.2 environment) =="
python evaluation/run_hpsv3.py --json_path $JsonPath --save_dir $SaveDir --model_path weights/MizzenAI__HPSv3/HPSv3.safetensors --batch_size 4
if ($LASTEXITCODE -ne 0) { Write-Warning "HPSv3 skipped: activate the mvte-hpsv3 environment and re-run this step" }

Write-Host "== 4/5 background FID / SSIM / PSNR / LPIPS =="
python evaluation/run_traditional.py --json_path $JsonPath --save_dir $SaveDir --batch_size $Batch
if ($LASTEXITCODE -ne 0) { throw "traditional metrics step failed" }

Write-Host "== 5/5 VLM judge (SC-T / SC-W / PQ); requires JUDGE_API_KEY =="
python evaluation/run_gpt_judgement.py --json_path $JsonPath --save_dir $SaveDir --batch_size 60
if ($LASTEXITCODE -ne 0) { throw "VLM judge step failed" }

Write-Host "== aggregating =="
python evaluation/huizong.py           --save_dir $SaveDir
python evaluation/huizong_rank.py      --save_dir $SaveDir --json_path $JsonPath
python evaluation/huizong_rank_2.py    --save_dir $SaveDir --json_path $JsonPath
python evaluation/huizong_language.py  --save_dir $SaveDir --json_path $JsonPath

Write-Host "done: $SaveDir/all_results.json"
