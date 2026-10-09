"""Judge-agnostic VLM evaluation for MVTE-benchmark (OpenAI-compatible endpoints).

Why this file exists separately from ``run_gpt_judgement.py``
------------------------------------------------------------
``run_gpt_judgement.py`` produced the numbers currently in the paper, via the
DashScope SDK hard-wired to ``qwen3-vl-flash``. It is kept untouched on purpose:
it is the reproducibility record for the submitted results. This script produces
*additional* results with a different (neutral) judge, into a separate directory,
so the two can be compared instead of one silently overwriting the other.

Output location
---------------
    <model_dir>/results_judge-<TAG>/
        gpt_target_area_semantic_accuracy.json   # same schema as before
        gpt_whole_semantic_accuracy.json         # same schema as before
        gpt_whole_perceptual_quality.json        # same schema as before
        judge_meta.json                          # judge id, prompt hash, params, failures
        failures.json                            # every sample that could not be scored
        state/<metric>.jsonl                     # resume checkpoint (one line per sample)

The three ``gpt_*.json`` filenames and schemas are unchanged on purpose, so
``huizong.py`` / ``get_xlsx.py`` can be pointed at the new directory as-is.

Fairness notes
--------------
* Cropped / masked judge inputs are REUSED from the original ``results/`` dir
  (``--crops_from``) rather than regenerated, so the new judge sees byte-identical
  pixels to the old one. Regeneration is only used as a fallback.
* Prompts are imported from ``prompts/`` unchanged.
* Every failure is recorded; nothing is silently dropped.

Usage
-----
    conda activate testval
    # smoke test: 8 samples of one metric on one model
    python run_vlm_judgement.py --model_dir ../model-output/ground-truth \
        --judge_model qwen3.8-flash --metrics sc_t --limit 8

    # full run
    python run_vlm_judgement.py --model_dir ../model-output/ground-truth \
        --judge_model kimi-k3
"""

import argparse
import base64
import concurrent.futures as cf
import hashlib
import io
import json
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import random
import re
import sys
import threading
import time
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from openai import OpenAI

from prompts.target_area_semantic_accuracy import *  # noqa: F401,F403
from prompts.whole_semantic_accuracy import *  # noqa: F401,F403
from prompts.whole_perceptual_quality import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATION_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))

DEFAULT_BASE_URL = "https://llm-dhfbjbolw1u66jy2.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #

def remap(p, data_root):
    """Map a stale absolute path from an older checkout onto data_root."""
    if not p:
        return p
    norm = p.replace("\\", "/")
    if os.path.exists(p):
        return p
    for marker in ("validation/",):
        i = norm.rfind(marker)
        if i >= 0:
            return os.path.join(data_root, *norm[i + len(marker):].split("/"))
    return os.path.join(data_root, *norm.split("/")[-3:])


def sanitize_tag(model_id):
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id)


# --------------------------------------------------------------------------- #
# image handling (identical geometry to the original pipeline)
# --------------------------------------------------------------------------- #

def mask_img(img_path, save_path, bbox_list):
    from PIL import Image, ImageDraw

    with Image.open(img_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        for bbox in bbox_list:
            draw.rectangle(bbox, fill="white", outline=None)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        img.save(save_path)


def crop_pad_and_resize(image_path, output_path, bbox, target_size=(512, 512)):
    from PIL import Image

    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    bbox = list(map(int, bbox))
    cropped = img.crop(bbox)
    w, h = cropped.size
    side = max(w, h)
    square = Image.new("RGB", (side, side), (255, 255, 255))
    square.paste(cropped, ((side - w) // 2, (side - h) // 2))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    square.resize(target_size, Image.Resampling.LANCZOS).save(output_path)


def to_data_url(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------- #
# response parsing (copied verbatim from run_gpt_judgement.py)
# --------------------------------------------------------------------------- #

def parse_evaluation_response(response_text):
    import ast

    default_result = {"score": -1}
    if not response_text:
        return default_result

    clean_text = re.sub(r"```json\s*|\s*```", "", response_text, flags=re.IGNORECASE)
    match = re.search(r"\{[\s\S]*\}", clean_text)
    if not match:
        return dict(default_result, error="no-json-block", raw=response_text[:400])

    json_str = match.group()
    parsed = None
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            return dict(default_result, error="json-parse-failed", raw=json_str[:400])

    if not isinstance(parsed, dict):
        return dict(default_result, error="not-a-dict")

    try:
        parsed["score"] = int(parsed.get("score", -1))
    except (ValueError, TypeError):
        parsed["score"] = -1
    return parsed


# --------------------------------------------------------------------------- #
# judge client
# --------------------------------------------------------------------------- #

class Judge:
    """Thin wrapper over an OpenAI-compatible chat endpoint with vision input."""

    def __init__(self, model, api_key, base_url, thinking=True, thinking_budget=81920,
                 temperature=None, seed=None, max_tokens=None, max_retries=6,
                 request_timeout=300):
        self.model = model
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             timeout=request_timeout, max_retries=0)
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "retries": 0, "hard_failures": 0}

    # ---- capability probe: some models reject enable_thinking / non-stream ----
    def probe(self):
        def attempt(thinking, stream):
            kw = dict(model=self.model,
                      messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                      stream=stream)
            if thinking:
                kw["extra_body"] = {"enable_thinking": True,
                                    "thinking_budget": self.thinking_budget}
            if stream:
                r = self.client.chat.completions.create(**kw)
                for _ in r:
                    pass
            else:
                self.client.chat.completions.create(**kw)

        for thinking in ([True, False] if self.thinking else [False]):
            for stream in ([True, False] if thinking else [False]):
                try:
                    attempt(thinking, stream)
                    self.thinking = thinking
                    self.stream = stream
                    return f"thinking={thinking} stream={stream}"
                except Exception as e:
                    last = f"thinking={thinking} stream={stream} -> {type(e).__name__}: {str(e)[:160]}"
        raise RuntimeError("judge probe failed for all modes: " + last)

    def _create(self, content):
        kw = dict(model=self.model,
                  messages=[{"role": "user", "content": content}],
                  stream=self.stream)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.seed is not None:
            kw["seed"] = self.seed
        if self.max_tokens is not None:
            kw["max_tokens"] = self.max_tokens
        if self.thinking:
            kw["extra_body"] = {"enable_thinking": True,
                                "thinking_budget": self.thinking_budget}
        return self.client.chat.completions.create(**kw)

    def ask(self, content):
        """content = list of {'type':'text'|'image_url', ...}. Returns parsed dict."""
        last_err = None
        for attempt in range(self.max_retries + 1):
            with self._lock:
                self.stats["calls"] += 1
                if attempt:
                    self.stats["retries"] += 1
            try:
                resp = self._create(content)
                if self.stream:
                    parts = []
                    for chunk in resp:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        c = getattr(delta, "content", None)
                        if c:
                            parts.append(c)
                    text = "".join(parts)
                else:
                    text = resp.choices[0].message.content or ""
                parsed = parse_evaluation_response(text)
                if parsed.get("score", -1) != -1:
                    return parsed
                # parsed but unusable -> retry, keep the reason
                parsed.setdefault("error", "score-missing-or-invalid")
                last_err = parsed
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt + random.random(), 30))
                    continue
                return last_err
            except Exception as e:  # network / rate limit / server error
                last_err = {"score": -1, "error": f"{type(e).__name__}: {str(e)[:300]}"}
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt + random.random(), 60))
                    continue
        with self._lock:
            self.stats["hard_failures"] += 1
        return last_err or {"score": -1, "error": "unknown"}


def build_content(items):
    """items = list of ('text', s) | ('image', path)."""
    content = []
    for kind, val in items:
        if kind == "text":
            content.append({"type": "text", "text": val})
        else:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(val)}})
    return content


# --------------------------------------------------------------------------- #
# task construction -- mirrors run_gpt_judgement.py exactly
# --------------------------------------------------------------------------- #

def build_sc_t_tasks(data, crops_dirs, roots):
    """Target-area semantic correctness (SC-T). 645 items.

    ``crops_dirs`` = (reuse_dir, fallback_dir). Reuse_dir is treated as READ-ONLY
    (it belongs to the original results/); any crop missing there is regenerated
    into fallback_dir. One such gap is known: font/00092_haibao.png is absent from
    full_train_8nodes_20260105_1/4000, which is why that model's SC-T in the paper
    averages 644 samples instead of 645.
    """
    reuse_dir, fallback_dir = crops_dirs
    ori_root, edit_root = roots
    tasks = []
    for key, items in data["data_list"].items():
        for item in items:
            fn, edit_prompt, edit_caption, sub, target, bboxes = item[0], item[1], item[2], item[3], item[4], item[5]
            ori_path = os.path.join(ori_root, key, fn)
            edit_path = os.path.join(edit_root, key, fn)

            def crop(subdir, src, bbox):
                if reuse_dir:
                    reused = os.path.join(reuse_dir, subdir, fn)
                    if os.path.exists(reused):
                        return reused
                out = os.path.join(fallback_dir, subdir, fn)
                if not os.path.exists(out):
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    crop_pad_and_resize(src, out, bbox)
                return out

            if sub == "add":
                p = crop(sub, edit_path, bboxes[0])
                tasks.append(dict(label=f"{sub}\\{fn}", imgs=[p],
                                  prompt=tasc_add.format(text=target),  # noqa: F405
                                  edit_subtype=sub, edit_prompt=edit_prompt))
            elif sub in ("color", "gradient", "texture"):
                p = crop(sub, edit_path, bboxes[0])
                tasks.append(dict(label=f"{sub}\\{fn}", imgs=[p],
                                  prompt=tasc_color_gradient_texture.format(text=target, statement=edit_caption),  # noqa: F405
                                  edit_subtype=sub, edit_prompt=edit_prompt))
            elif sub == "font":
                p = crop(sub, edit_path, bboxes[0])
                tasks.append(dict(label=f"{sub}\\{fn}", imgs=[p],
                                  prompt=tasc_font.format(text=target, statement=edit_caption),  # noqa: F405
                                  edit_subtype=sub, edit_prompt=edit_prompt))
            elif sub == "correct":
                p = crop(sub, edit_path, bboxes[0])
                tasks.append(dict(label=f"{sub}\\{fn}", imgs=[p],
                                  prompt=tasc_correct.format(text=target.split("-")[1]),  # noqa: F405
                                  edit_subtype=sub, edit_prompt=edit_prompt))
            elif sub == "edit":
                p = crop(sub, edit_path, bboxes[0])
                tasks.append(dict(label=f"{sub}\\{fn}", imgs=[p],
                                  prompt=tasc_edit.format(text_1=target.split("-")[0], text_2=target.split("-")[1]),  # noqa: F405
                                  edit_subtype=sub, edit_prompt=edit_prompt))
            elif sub in ("weight", "size"):
                po = crop(sub + "_old", ori_path, bboxes[0])
                pn = crop(sub + "_new", edit_path, bboxes[0])
                tasks.append(dict(label=f"{sub}\\{fn}",
                                  imgs=[("The Reference Image", po), ("The Edited Image", pn)],
                                  prompt=tasc_weight_size.format(text=target, instruction=edit_prompt),  # noqa: F405
                                  edit_subtype=sub, edit_prompt=edit_prompt))
            elif sub == "remove":
                p = crop(sub, edit_path, bboxes[0])
                tasks.append(dict(label=f"{sub}\\{fn}", imgs=[p],
                                  prompt=tasc_remove.format(text_to_remove=target),  # noqa: F405
                                  edit_subtype=sub, edit_prompt=edit_prompt))
            elif sub == "rotate":
                p = crop(sub, edit_path, bboxes[0])
                tasks.append(dict(label=f"{sub}\\{fn}", imgs=[p],
                                  prompt=tasc_rotate.format(text=target, description=edit_caption),  # noqa: F405
                                  edit_subtype=sub, edit_prompt=edit_prompt))
            elif sub == "move":
                p_rem = crop("move_remove", edit_path, bboxes[0])
                tasks.append(dict(label=f"move_remove\\{fn}", imgs=[p_rem],
                                  prompt=tasc_remove.format(text_to_remove=target),  # noqa: F405
                                  edit_subtype="move_remove", edit_prompt=edit_prompt))
                p_add = crop("move_add", edit_path, bboxes[1])
                tasks.append(dict(label=f"move_add\\{fn}", imgs=[p_add],
                                  prompt=tasc_add.format(text=target),  # noqa: F405
                                  edit_subtype="move_add", edit_prompt=edit_prompt))
            else:
                raise ValueError(f"unknown edit_subtype: {sub}")
    return tasks


def build_sc_w_tasks(data, roots):
    """Whole-image semantic correctness (SC-W). 545 items."""
    ori_root, edit_root = roots
    tpl = {
        "add": lambda t, c, s, ip: wsc_add.format(text=t, instruction=ip),  # noqa: F405
        "color": lambda t, c, s, ip: wsc_color_gradient_texture.format(text=t, instruction=ip),  # noqa: F405
        "gradient": lambda t, c, s, ip: wsc_color_gradient_texture.format(text=t, instruction=ip),  # noqa: F405
        "texture": lambda t, c, s, ip: wsc_color_gradient_texture.format(text=t, instruction=ip),  # noqa: F405
        "font": lambda t, c, s, ip: wsc_font.format(text=t, instruction=ip),  # noqa: F405
        "correct": lambda t, c, s, ip: wsc_correct.format(instruction=ip, original_text=t.split("-")[0], target_text=t.split("-")[1]),  # noqa: F405
        "edit": lambda t, c, s, ip: wsc_edit.format(text_1=t.split("-")[0], text_2=t.split("-")[1], instruction=ip),  # noqa: F405
        "weight": lambda t, c, s, ip: wsc_weight_size.format(text=t, instruction=ip),  # noqa: F405
        "size": lambda t, c, s, ip: wsc_weight_size.format(text=t, instruction=ip),  # noqa: F405
        "remove": lambda t, c, s, ip: wsc_remove.format(text_to_remove=t, instruction=ip),  # noqa: F405
        "rotate": lambda t, c, s, ip: wsc_rotate.format(text=t, instruction=ip),  # noqa: F405
        "move": lambda t, c, s, ip: wsc_move.format(text_to_move=t, editing_instruction=ip),  # noqa: F405
    }
    tasks = []
    for key, items in data["data_list"].items():
        for item in items:
            fn, edit_prompt, edit_caption, sub, target = item[0], item[1], item[2], item[3], item[4]
            ori_path = os.path.join(ori_root, key, fn)
            edit_path = os.path.join(edit_root, key, fn)
            if sub not in tpl:
                raise ValueError(f"unknown edit_subtype: {sub}")
            tasks.append(dict(label=f"{sub}\\{fn}",
                              imgs=[("The Original Image", ori_path), ("The Edited Image", edit_path)],
                              prompt=tpl[sub](target, edit_caption, sub, edit_prompt),
                              edit_subtype=sub, edit_prompt=edit_prompt))
    return tasks


def build_pq_tasks(data, masks_dirs, roots):
    """Background perceptual quality (PQ). 545 items, bbox regions painted white.

    ``masks_dirs`` = (reuse_dir, fallback_dir); reuse_dir stays read-only.
    """
    reuse_dir, fallback_dir = masks_dirs
    ori_root, edit_root = roots
    tasks = []
    for key, items in data["data_list"].items():
        for item in items:
            fn, edit_prompt, sub, bboxes = item[0], item[1], item[3], item[5]
            ori_path = os.path.join(ori_root, key, fn)
            edit_path = os.path.join(edit_root, key, fn)

            def resolve(kind, src):
                if reuse_dir:
                    reused = os.path.join(reuse_dir, kind, sub, fn)
                    if os.path.exists(reused):
                        return reused
                out = os.path.join(fallback_dir, kind, sub, fn)
                if not os.path.exists(out):
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    mask_img(src, out, bboxes)
                return out

            tasks.append(dict(label=f"{sub}\\{fn}",
                              imgs=[("The Original Image", resolve("original", ori_path)),
                                    ("The Compressed Image", resolve("edited", edit_path))],
                              prompt=wpq,  # noqa: F405
                              edit_subtype=sub, edit_prompt=edit_prompt))
    return tasks


def task_content(task):
    content = [{"type": "text", "text": task["prompt"]}]
    for entry in task["imgs"]:
        if isinstance(entry, tuple):
            caption, path = entry
            content.append({"type": "text", "text": caption})
            content.append({"type": "image_url", "image_url": {"url": to_data_url(path)}})
        else:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(entry)}})
    return content


# --------------------------------------------------------------------------- #
# runner with resume + incremental checkpoint
# --------------------------------------------------------------------------- #

def run_metric(judge, tasks, state_path, concurrency, limit=None, label=""):
    done = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("score", -1) != -1:
                    done[rec["label"]] = rec
    todo = [t for t in tasks if t["label"] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"  [{label}] total={len(tasks)} cached={len(done)} todo={len(todo)}")

    written = 0
    if todo:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        wlock = threading.Lock()
        fh = open(state_path, "a", encoding="utf-8")
        t0 = time.time()

        def work(task):
            res = judge.ask(task_content(task))
            rec = {"label": task["label"], "edit_subtype": task["edit_subtype"],
                   "edit_prompt": task["edit_prompt"], "prompt_sha1": hashlib.sha1(
                       task["prompt"].encode("utf-8")).hexdigest()[:12]}
            rec.update(res)
            with wlock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
            return rec

        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for i, fut in enumerate(cf.as_completed(futs), 1):
                rec = fut.result()
                if rec.get("score", -1) != -1:
                    done[rec["label"]] = rec
                if i % 50 == 0 or i == len(futs):
                    el = time.time() - t0
                    print(f"    {i}/{len(futs)}  {el / i:.2f}s/call  "
                          f"eta={el / i * (len(futs) - i) / 60:.1f}min", flush=True)
        fh.close()
        written = len(todo)

    # keep the ORIGINAL task order so downstream index alignment matches the old files
    ordered = []
    for t in tasks:
        rec = done.get(t["label"])
        if rec is None:
            continue
        ordered.append({"label": t["label"], "img_path": t["imgs"][0] if isinstance(t["imgs"][0], str) else t["imgs"][0][1],
                        "prompt": t["prompt"], "edit_subtype": t["edit_subtype"],
                        "edit_prompt": t["edit_prompt"],
                        "gpt_judgement": {k: v for k, v in rec.items()
                                          if k not in ("label", "edit_subtype", "edit_prompt", "prompt_sha1")}})
    missing = [t["label"] for t in tasks if t["label"] not in done]
    return ordered, missing, written


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="e.g. model_outputs/<your_model>")
    ap.add_argument("--judge_model", default="qwen3.8-flash")
    ap.add_argument("--base_url", default=os.getenv("JUDGE_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--api_key", default=os.getenv("JUDGE_API_KEY"))
    ap.add_argument("--out_suffix", default=None,
                    help="output dir = results_judge-<judge_model>[-<suffix>]")
    ap.add_argument("--data_root", default=VALIDATION_ROOT)
    ap.add_argument("--crops_from", default=None,
                    help="existing results/ dir to reuse crops+mask images from "
                         "(default: <model_dir>/results)")
    ap.add_argument("--metrics", default="sc_t,sc_w,pq")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="per-metric smoke-test cap")
    ap.add_argument("--thinking", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--thinking_budget", type=int, default=81920)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_retries", type=int, default=6)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("set the DASHSCOPE_API_KEY environment variable, or pass --api_key")

    model_dir = os.path.abspath(args.model_dir)
    json_path = os.path.join(model_dir, "validation_filter.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(model_dir, "validation.json")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        for _k in ("original_imgs_root", "edited_imgs_root"):
            if isinstance(data.get(_k), str) and not os.path.isabs(data[_k]):
                data[_k] = os.path.join(REPO_ROOT, data[_k])

    roots = (remap(data["original_imgs_root"], args.data_root),
             remap(data["edited_imgs_root"], args.data_root))
    print("model_dir :", model_dir)
    print("originals :", roots[0], os.path.isdir(roots[0]))
    print("edited    :", roots[1], os.path.isdir(roots[1]))
    if not (os.path.isdir(roots[0]) and os.path.isdir(roots[1])):
        raise SystemExit("image roots not found -- pass --data_root explicitly")

    tag = sanitize_tag(args.judge_model) + (f"-{args.out_suffix}" if args.out_suffix else "")
    out_dir = os.path.join(model_dir, f"results_judge-{tag}")
    state_dir = os.path.join(out_dir, "state")
    os.makedirs(state_dir, exist_ok=True)

    crops_src = args.crops_from or os.path.join(model_dir, "results")
    reuse_tasa = os.path.join(crops_src, "tmp_tasa_dir")
    reuse_wpq = os.path.join(crops_src, "tmp_wpq_dir")
    if not os.path.isdir(reuse_tasa):
        reuse_tasa = None
    if not os.path.isdir(reuse_wpq):
        reuse_wpq = None
    # (read-only reuse dir, writable fallback dir)
    tasa_dirs = (reuse_tasa, os.path.join(out_dir, "tmp_tasa_dir"))
    wpq_dirs = (reuse_wpq, os.path.join(out_dir, "tmp_wpq_dir"))
    print("crops from:", reuse_tasa or "(regenerate)")
    print("masks from:", reuse_wpq or "(regenerate)")
    print("out dir   :", out_dir)

    judge = Judge(args.judge_model, args.api_key, args.base_url,
                  thinking=(True if args.thinking == "auto" else args.thinking == "on"),
                  thinking_budget=args.thinking_budget,
                  temperature=args.temperature, seed=args.seed,
                  max_retries=args.max_retries)
    mode = judge.probe()
    print("judge mode:", mode)

    wanted = {m.strip() for m in args.metrics.split(",") if m.strip()}
    builders = {
        "sc_t": ("gpt_target_area_semantic_accuracy.json",
                 lambda: build_sc_t_tasks(data, tasa_dirs, roots)),
        "sc_w": ("gpt_whole_semantic_accuracy.json",
                 lambda: build_sc_w_tasks(data, roots)),
        "pq": ("gpt_whole_perceptual_quality.json",
               lambda: build_pq_tasks(data, wpq_dirs, roots)),
    }

    failures = {}
    started = datetime.now().isoformat(timespec="seconds")
    for key in ("sc_t", "sc_w", "pq"):
        if key not in wanted:
            continue
        fname, build = builders[key]
        print(f"\n=== metric {key} -> {fname}")
        tasks = build()
        ordered, missing, _ = run_metric(
            judge, tasks, os.path.join(state_dir, f"{key}.jsonl"),
            args.concurrency, args.limit, label=key)
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=4, ensure_ascii=False)
        if missing:
            failures[key] = missing
            print(f"  !! {len(missing)} samples unscored, e.g. {missing[:3]}")

    prompt_files = [os.path.join(HERE, "prompts", n) for n in
                    ("target_area_semantic_accuracy.py", "whole_semantic_accuracy.py",
                     "whole_perceptual_quality.py")]
    h = hashlib.sha256()
    for pf in prompt_files:
        if os.path.exists(pf):
            h.update(open(pf, "rb").read())

    meta = {
        "judge_model": args.judge_model,
        "judge_base_url": args.base_url,
        "judge_mode": mode,
        "temperature": args.temperature,
        "seed": args.seed,
        "thinking_budget": args.thinking_budget if judge.thinking else None,
        "prompts_sha256": h.hexdigest(),
        "model_dir": model_dir,
        "json_path": json_path,
        "crops_reused_from": crops_src,
        "original_results_dir": os.path.join(model_dir, "results"),
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "api_stats": judge.stats,
        "unscored": failures,
        "metrics_requested": sorted(wanted),
        "note": "VLM judge re-score. Non-VLM metrics (OCR/SigLIP2/HPSv3/FID/SSIM/PSNR) "
                "are unchanged and must be merged in from original_results_dir.",
    }
    with open(os.path.join(out_dir, "judge_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    if failures:
        with open(os.path.join(out_dir, "failures.json"), "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=4, ensure_ascii=False)

    print("\njudge stats:", judge.stats)
    print("wrote:", out_dir)


if __name__ == "__main__":
    main()
