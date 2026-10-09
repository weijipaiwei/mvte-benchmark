# MVTE-benchmark and its evaluation harness

MVTE-benchmark is a 545-sample benchmark for **instruction-driven visual text
editing under a mask-free protocol**: the model receives only the source image
and an editing *action* instruction (add / delete / replace / correct / recolor /
retexture / regradient / restyle / reweight / resize / move / rotate) and must
execute it without any test-time mask, box or layout condition.

Each sample provides

* the source image (`benchmark/images/original/<group>/<file>`),
* the ground-truth edited image (`benchmark/images/ground_truth_edited/<group>/<file>`),
* a hand-written editing action instruction,
* a description of the post-edit text state,
* the exact text bounding box(es),
* the editing sub-type and a human difficulty rating
  (`length / area / visual_around / font`, each 1-3),

all in `benchmark/validation.json`.

Sub-task counts: add 100, color 33, gradient 37, texture 30, font 10, edit 100,
weight 10, size 10, correct 5, move 100, rotate 10, remove 100.

## Repository layout

```
benchmark/
    validation.json                  annotations for the 545 samples
    images/original/                 source images
    images/ground_truth_edited/      ground-truth edited images
evaluation/
    run_ocr.py                       ACC / NED with DeepSeek-OCR on the cropped target region
    run_siglip2.py                   SigLIP2 image-image / image-text scores
    run_hpsv3.py                     HPSv3 whole-image preference score
    run_traditional.py               FID / SSIM / PSNR / LPIPS on the masked background
    run_gpt_judgement.py             VLM judge: SC-T, SC-W, PQ (DashScope SDK)
    run_vlm_judgement.py             same three metrics through any OpenAI-compatible
                                     VLM endpoint (used for neutral-judge re-scoring;
                                     resumable, thread-pooled)
    filter_data.py                   intersect a model output json with the benchmark json
    huizong*.py                      per-subtask aggregation of the seven metric files
    get_xlsx*.py                     export of the aggregated tables to Excel
    prompts/                         the exact judge prompts (English and Chinese PQ variants)
    run_all.sh / run_all.ps1         one-command evaluation of a single model
```

## Setup

```bash
conda create -n mvte python=3.10 -y && conda activate mvte
pip install -r requirements.txt
```

Model weights are **not** included. Download them into `weights/` (or pass
`--model_path` explicitly):

```bash
huggingface-cli download deepseek-ai/DeepSeek-OCR               --local-dir weights/deepseek-ai__DeepSeek-OCR
huggingface-cli download google/siglip2-so400m-patch16-naflex   --local-dir weights/google__siglip2-so400m-patch16-naflex
huggingface-cli download MizzenAI/HPSv3                         --local-dir weights/MizzenAI__HPSv3
```

> `run_siglip2.py` needs `transformers>=4.57` while `run_hpsv3.py` needs
> `transformers==4.45.2`. `requirements-hpsv3.txt` pins the older stack; install
> it in a second environment if you want to run both without switching by hand.

The VLM judge needs an API key:

```bash
export DASHSCOPE_API_KEY=...        # for run_gpt_judgement.py / run_vlm_judgement.py
```

No key is stored anywhere in this repository.

## Evaluating your own model

1. Render/edit the 545 source images, keeping the folder structure
   `model_outputs/<your_model>/edited/<group>/<file>`.
2. Create `model_outputs/<your_model>/validation.json` with the same schema as
   `benchmark/validation.json` but `edited_imgs_root` pointing at your folder.
3. Intersect with the benchmark (drops samples you failed to produce, and is
   reported in the paper whenever it happens):

   ```bash
   python evaluation/filter_data.py \
       --json_path1 benchmark/validation.json \
       --json_path2 model_outputs/<your_model>/validation.json
   ```

4. Run everything:

   ```bash
   bash evaluation/run_all.sh model_outputs/<your_model>/validation.json \
                              model_outputs/<your_model>/results
   ```

   or the individual runners, e.g.

   ```bash
   python evaluation/run_ocr.py          --json_path <json> --save_dir <out> --ocr_name deepseek-ocr
   python evaluation/run_siglip2.py      --json_path <json> --save_dir <out> --model_path weights/google__siglip2-so400m-patch16-naflex
   python evaluation/run_hpsv3.py        --json_path <json> --save_dir <out> --model_path weights/MizzenAI__HPSv3/HPSv3.safetensors
   python evaluation/run_traditional.py  --json_path <json> --save_dir <out>
   python evaluation/run_gpt_judgement.py --json_path <json> --save_dir <out> --batch_size 60
   ```

5. Aggregate:

   ```bash
   python evaluation/huizong.py        --save_dir <out>
   python evaluation/huizong_rank.py   --save_dir <out> --json_path <json>
   python evaluation/huizong_rank_2.py --save_dir <out> --json_path <json>
   python evaluation/huizong_language.py --save_dir <out> --json_path <json>
   ```

All scripts resolve the image roots stored in the json relative to the
repository root, so they can be run from anywhere.

## Metric definitions

| metric | what it measures | region |
| --- | --- | --- |
| ACC / NED | OCR sentence accuracy / normalised edit distance vs the target string | cropped target region |
| S-I2I | SigLIP2 image-image similarity between edited and ground-truth crops | cropped target region |
| SC-T | VLM judge: is the target text edited as instructed? (1/3/5/7/9) | cropped target region |
| SC-W | VLM judge: was the instruction executed on the whole image? (1/3/5/7/9) | full image pair |
| PQ | VLM judge: is the masked background unchanged? (1/3/5/7/9) | full image pair, bbox painted white |
| FID / SSIM / PSNR / LPIPS | background preservation | full image, bbox painted white |
| HPSv3 | human-preference proxy | full edited image |

For `remove` and `move_old` the OCR score is inverted when averaged: there the
*absence* of the target string is the correct outcome.

The judge prompts are released verbatim in `evaluation/prompts/` so that scores
can be reproduced or re-judged with a different VLM; `run_vlm_judgement.py`
reuses the cropped/masked judge inputs of a previous run byte-for-byte, which
makes judge-swap comparisons fair.

## License

This repository is dual-licensed, and the scope of each licence is fixed by its
location:

* **Code** (`evaluation/`, the drivers, and everything else outside
  `benchmark/`) -- Apache License 2.0, see [`LICENSE`](LICENSE).
* **Data** (`benchmark/`: the 545-sample annotation file, the source images and
  the ground-truth edited images) -- Creative Commons Attribution 4.0
  International, see [`benchmark/LICENSE`](benchmark/LICENSE).

Both licences permit free academic and commercial use; CC-BY-4.0 additionally
requires attribution, which is satisfied by citing the paper below.

Copyright 2026 Chenfeng Zhang. This notice covers both the code and the data
released here; the Apache-2.0 appendix in `LICENSE` carries the same holder.

## Citation

TODO(before upload): insert the BibTeX of the accompanying paper once the
journal version is accepted. Until then, the attribution required by CC-BY-4.0
is satisfied by citing the ACM MM 2026 submission of this work.
