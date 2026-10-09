#!/usr/bin/env bash
# Evaluate one model on MVTE-benchmark.
#
#   bash evaluation/run_all.sh <model_validation.json> <save_dir> [batch_size]
#
# Example:
#   bash evaluation/run_all.sh model_outputs/my_model/validation.json \
#                              model_outputs/my_model/results
#
# Run `filter_data.py` first (see README) so that <model_validation.json> only
# contains samples that were actually produced.
set -euo pipefail

JSON_PATH="${1:?usage: run_all.sh <model_validation.json> <save_dir> [batch_size]}"
SAVE_DIR="${2:?usage: run_all.sh <model_validation.json> <save_dir> [batch_size]}"
BATCH="${3:-32}"

# always run from the repository root so the relative image roots resolve
cd "$(dirname "$0")/.."

echo "== 1/5 OCR (ACC / NED) =="
python evaluation/run_ocr.py --json_path "$JSON_PATH" --save_dir "$SAVE_DIR" --ocr_name deepseek-ocr

echo "== 2/5 SigLIP2 =="
python evaluation/run_siglip2.py --json_path "$JSON_PATH" --save_dir "$SAVE_DIR" \
    --model_path weights/google__siglip2-so400m-patch16-naflex

echo "== 3/5 HPSv3 (needs the transformers==4.45.2 environment) =="
python evaluation/run_hpsv3.py --json_path "$JSON_PATH" --save_dir "$SAVE_DIR" \
    --model_path weights/MizzenAI__HPSv3/HPSv3.safetensors --batch_size 4 || \
    echo "!! HPSv3 skipped: activate the mvte-hpsv3 environment and re-run this step"

echo "== 4/5 background FID / SSIM / PSNR / LPIPS =="
python evaluation/run_traditional.py --json_path "$JSON_PATH" --save_dir "$SAVE_DIR" --batch_size "$BATCH"

echo "== 5/5 VLM judge (SC-T / SC-W / PQ); requires JUDGE_API_KEY =="
python evaluation/run_gpt_judgement.py --json_path "$JSON_PATH" --save_dir "$SAVE_DIR" --batch_size 60

echo "== aggregating =="
python evaluation/huizong.py          --save_dir "$SAVE_DIR"
python evaluation/huizong_rank.py     --save_dir "$SAVE_DIR" --json_path "$JSON_PATH"
python evaluation/huizong_rank_2.py   --save_dir "$SAVE_DIR" --json_path "$JSON_PATH"
python evaluation/huizong_language.py --save_dir "$SAVE_DIR" --json_path "$JSON_PATH"

echo "done: $SAVE_DIR/all_results.json"
