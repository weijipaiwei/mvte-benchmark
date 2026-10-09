
import torch
import torch.nn.functional as F
import json
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import argparse
from PIL import Image
from transformers.image_utils import load_image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import random
import numpy as np
from typing import Tuple
import re
import sys
from tqdm import tqdm
import time

def seed_everything(seed=42):
    """
    Fix every possible random seed for reproducibility.
    """
    random.seed(seed)                 # Python's built-in random
    os.environ['PYTHONHASHSEED'] = str(seed) # disable hash randomisation
    np.random.seed(seed)              # numpy random
    torch.manual_seed(seed)           # torch cpu
    torch.cuda.manual_seed(seed)      # torch gpu
    torch.cuda.manual_seed_all(seed)  # torch multi-gpu
    
    # --- strict mode (optional) ---
    # costs a little performance but keeps the convolutions deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class deepseek_ocr:
    def __init__(self, model_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
        self.model = self.model.eval().cuda().to(torch.bfloat16)
    def get_results(self,img_path, tmp_output_path):
        prompt = "<image>\nFree OCR. "
        t1 = time.time()
        res = self.model.infer(self.tokenizer, prompt=prompt, image_file=img_path, output_path = tmp_output_path, base_size = 1024, image_size = 1024, crop_mode=False, save_results = True, test_compress = True)
        
        with open(os.path.join(tmp_output_path, 'result.mmd'), 'r', encoding='utf-8') as f:
            content = f.readlines()
        content = [i.strip().replace("**", '') for i in content if i.strip()]
        return content





def find_best_matching_substring(text: str, pattern: str) -> Tuple[str, float]:
    """
    Find the substring of text with the smallest edit distance to pattern; return it with its normalised edit distance.

    Args:
        text (str): haystack string (A)
        pattern (str): target string (B)

    Returns:
        Tuple[str, float]: 
            - the best-matching substring
            - normalised edit distance (0.0 = exact match, 1.0 = total mismatch)
    """
    
    # --- 1. robustness checks ---
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("input must be a string")
        
    m, n = len(text), len(pattern)
    
    # edge-case handling
    if n == 0:
        # empty pattern: matches any empty substring, distance 0
        return "", 0.0
    if m == 0:
        # empty haystack: no match possible, distance maximal (return "" and 1.0 for robustness)
        return "", 1.0

    # --- 2. initialise the DP matrix ---
    # dp[i][j] = min edit distance between pattern[:i] and text[:j]
    # int32/int would save memory, but a Python list is fine
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # first column: shrinking the pattern to empty costs i deletions
    for i in range(n + 1):
        dp[i][0] = i
        
    # first row: an empty pattern matches any prefix of text at cost 0
    # this allows "semi-global alignment": leading text characters are skipped for free
    for j in range(m + 1):
        dp[0][j] = 0

    # --- 3. fill the DP matrix ---
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pattern[i - 1] == text[j - 1] else 1
            
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion (from pattern)
                dp[i][j - 1] + 1,      # insertion (into pattern)
                dp[i - 1][j - 1] + cost # substitute / match
            )

    # --- 4. find the best end position ---
    # take the minimum of the last row: trailing text characters may be skipped for free.
    raw_min_dist = float('inf')
    end_idx = -1
    
    for j in range(m + 1):
        if dp[n][j] < raw_min_dist:
            raw_min_dist = dp[n][j]
            end_idx = j
            
    if end_idx == -1:
        return "", 1.0

    # --- 5. backtrack to the start position ---
    curr_i, curr_j = n, end_idx
    
    while curr_i > 0:
        cost = 0 if pattern[curr_i - 1] == text[curr_j - 1] else 1
        
        # priority: match/substitute > above > left
        # this priority favours a more compact alignment and avoids needless expansion
        if curr_j > 0 and dp[curr_i][curr_j] == dp[curr_i - 1][curr_j - 1] + cost:
            curr_i -= 1
            curr_j -= 1
        elif dp[curr_i][curr_j] == dp[curr_i - 1][curr_j] + 1:
            curr_i -= 1
        else:
            curr_j -= 1
            
    start_idx = curr_j
    
    # extract the substring
    matched_substring = text[start_idx:end_idx]
    
    # --- 6. compute the normalised edit distance ---
    # the normalisation denominator is usually the longer of the two strings
    # here we compare pattern with the matched_substring we found
    max_len = max(len(pattern), len(matched_substring))
    
    if max_len == 0:
        normalized_dist = 0.0
    else:
        normalized_dist = raw_min_dist / max_len

    return matched_substring, normalized_dist


def normalize_punctuation(text: str) -> str:
    """
    Convert all Chinese (full-width) punctuation in the text to the ASCII equivalents.
    
    Args:
        text (str): input string containing Chinese punctuation.
        
    Returns:
        str: the converted string; TypeError if the input is not a string.
    """
    
    # 1. robustness check: input must be a string
    if not isinstance(text, str):
        # return str(text) is an alternative forced conversion, if needed
        raise TypeError(f"Input must be a string, got {type(text)}")

    # 2. define the mapping
    # single-character mapping (the fast part)
    # covers common punctuation, brackets, quotes and the full-width space
    chinese_chars = "，。！？：；“”‘’（）【】《》｛｝、|~`@#￥%&*=+\u3000"  # \u3000 is the full-width space
    english_chars = ",.!?:;\"\"''()[]<>{}||~`@#$%&*=  " # the corresponding ASCII symbols
    
    # build the translation table
    trans_table = str.maketrans(chinese_chars, english_chars)
    
    # 3. apply the single-character mapping
    result = text.translate(trans_table)
    
    # 4. handle multi-character symbols (translate cannot do 1-to-many or many-to-1)
    # the Chinese ellipsis is two chars: "……" -> "..."
    # the Chinese dash is two chars: "——" -> "--"
    special_replacements = {
        "……": "...",
        "——": "--",
    }
    
    for cn, en in special_replacements.items():
        result = result.replace(cn, en)
        
    return result


def crop_image(img_path, bbox):
    try:
        img = Image.open(img_path)
        img = img.convert('RGB')
        # make sure the bbox coordinates are integers
        bbox = [int(x) for x in bbox]
        cropped_img = img.crop(bbox)
        return cropped_img
    except Exception as e:
        print(f"Error cropping {img_path}: {e}")
        return Image.new('RGB', (100, 100), (0, 0, 0)) # black placeholder to avoid aborting

def pad_image_to_square(input_image):
    """
    Pad the image to a square without changing the longest side.
    Paint white and force RGB.
    Pad to 256x256 if the final square is smaller.
    """
    # 1. force RGB (and handle the alpha channel)
    if input_image.mode in ("RGBA", "P"):
        # create a white canvas the size of the source image to handle transparency
        temp_bg = Image.new("RGB", input_image.size, (255, 255, 255))
        # for RGBA, paste using the alpha channel as mask
        if input_image.mode == "RGBA":
            temp_bg.paste(input_image, mask=input_image.split()[3])
        else:
            temp_bg.paste(input_image)
        rgb_img = temp_bg
    else:
        rgb_img = input_image.convert("RGB")

    # 2. read the source size and pick the square side
    width, height = rgb_img.size
    # [change] take the max of width, height and 256
    # keeps the image square and at least 256 px
    side_length = max(width, height, 512)
    
    # 3. create a white square canvas
    square_img = Image.new("RGB", (side_length, side_length), (255, 255, 255))
    
    # 4. compute the centring offset
    # logic unchanged: centred paste whether padding to a square or to 256
    left = (side_length - width) // 2
    top = (side_length - height) // 2
    
    # 5. paste the source image
    square_img.paste(rgb_img, (left, top))
    
    return square_img


def get_acc_ned(text_list, text):
    text_list = [normalize_punctuation(i) for i in text_list]
    text = normalize_punctuation(text)

    min_ned = 1e10
    best_match = ""
    for t in text_list:
        match, ned = find_best_matching_substring(t, text)
        if ned < min_ned:
            min_ned = ned
            best_match = match
    
    if best_match == text:
        acc = 1
    else:
        acc = 0

    return acc, min_ned, best_match



def calc_ocr_ned(data_info, ocr_model, model_name, tmp_dir):
    edited_imgs_root = data_info['edited_imgs_root']
    data_list = data_info['data_list']

    label_list = []
    edited_img_target_area_path = []
    target_text = []

    for key in data_list.keys():
        for item in data_list[key]:
            edited_img_path = os.path.join(edited_imgs_root, key, item[0])
            tmp_edited_path = os.path.join(tmp_dir, key, item[0])
            prompt = item[1]
            caption = item[2]
            edit_subtype = item[3]
            edit_target = item[4]
            bbox_list = item[5]
            comment = item[6]

            if edit_subtype in ['add', 'remove']:
                label_list.append(edit_subtype + '\\' + item[0])
                os.makedirs(os.path.dirname(tmp_edited_path), exist_ok=True)
                if not os.path.exists(tmp_edited_path):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[0]))
                    cropped_image.save(tmp_edited_path)
                edited_img_target_area_path.append(tmp_edited_path)
                target_text.append(edit_target)

            elif edit_subtype in ['edit', 'correct']:
                label_list.append(edit_subtype + '\\' + item[0])
                os.makedirs(os.path.dirname(tmp_edited_path), exist_ok=True)
                if not os.path.exists(tmp_edited_path):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[0]))
                    cropped_image.save(tmp_edited_path)
                edited_img_target_area_path.append(tmp_edited_path)
                target_text.append(edit_target.split('-')[1])

            elif edit_subtype == 'move':
                label_list.append(edit_subtype + '_old\\' + item[0])
                tmp_edited_path_old = tmp_edited_path.replace(key, key+'_old')
                os.makedirs(os.path.dirname(tmp_edited_path_old), exist_ok=True)
                if not os.path.exists(tmp_edited_path_old):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[0]))
                    cropped_image.save(tmp_edited_path_old)
                edited_img_target_area_path.append(tmp_edited_path_old)
                target_text.append(edit_target)

                label_list.append(edit_subtype + '_new\\' + item[0])
                tmp_edited_path_new = tmp_edited_path.replace(key, key+'_new')
                os.makedirs(os.path.dirname(tmp_edited_path_new), exist_ok=True)
                if not os.path.exists(tmp_edited_path_new):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[1]))
                    cropped_image.save(tmp_edited_path_new)
                edited_img_target_area_path.append(tmp_edited_path_new)
                target_text.append(edit_target)

            else:
                continue

    result_list = []
    mean_acc = 0
    mean_ned = 0
    minus = 0
    for i in tqdm(range(len(label_list))):
        if model_name == 'paddle-ocr':
            output = ocr_model.predict(edited_img_target_area_path[i])[0]._to_json()['res']['parsing_res_list']
            ocr_texts = [i['block_content'].strip().lower() for i in output]
        elif model_name == 'deepseek-ocr':
            ocr_tmp_dir = os.path.join(tmp_dir, 'tmp_result', label_list[i].replace('\\', '_').split('.')[0])
            ocr_texts = ocr_model.get_results(edited_img_target_area_path[i], ocr_tmp_dir)
            ocr_texts = [i.lower() for i in ocr_texts]
        else:
            raise

        # TODO test whether the text list needs to be concatenated
        acc, ned, best_match = get_acc_ned(ocr_texts, target_text[i].lower())
        if acc != 1:
            acc_, ned_, best_match_ = get_acc_ned(["".join(ocr_texts)], target_text[i].lower())
            if ned_ < ned:
                acc = acc_
                ned = ned_
                best_match = best_match_

        edit_type = label_list[i].split('\\')[0]
        if ned > 1:
            ned = 1
        if edit_type == 'remove' or edit_type == 'move_old':
            minus += 1
        else:
            mean_acc += acc
            mean_ned += ned

        result_list.append({'acc': acc, 'ned': ned, 'best_match': best_match, 'target': target_text[i], "label": label_list[i], "candidates": ocr_texts})
    
    mean_acc /= (len(label_list) - minus)
    mean_ned /= (len(label_list) - minus)
    return {"mean_acc": mean_acc, "mean_ned": mean_ned, "label_list": label_list, 'per_image_result': result_list}



def run_evaluation(data, save_dir, ocr_model, model_name):
    os.makedirs(save_dir, exist_ok=True)
    tmp_dir = os.path.join(save_dir, 'ocr_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    output = calc_ocr_ned(data, ocr_model, model_name, tmp_dir)

    with open(os.path.join(save_dir, f"{model_name}_acc_ned.json"), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    seed_everything(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default=os.path.join(REPO_ROOT, "benchmark", "validation.json"), help="path of the benchmark json")
    parser.add_argument('--save_dir', type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument('--ocr_name', type=str, default="deepseek-ocr", help = "paddle-ocr or deepseek-ocr")
    args = parser.parse_args()

    # check that the path exists
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        exit()

    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for _k in ("original_imgs_root", "edited_imgs_root"):
            if isinstance(data.get(_k), str) and not os.path.isabs(data[_k]):
                data[_k] = os.path.join(REPO_ROOT, data[_k])

    save_dir = args.save_dir
    if args.ocr_name == 'paddle-ocr':
        from paddleocr import PaddleOCRVL
        ocr_model = PaddleOCRVL()
    elif args.ocr_name == 'deepseek-ocr':
        model_name = os.path.join(REPO_ROOT, "weights", "deepseek-ai__DeepSeek-OCR")
        ocr_model = deepseek_ocr(model_name)
    else:
        raise
    

    run_evaluation(data, save_dir, ocr_model, args.ocr_name)
