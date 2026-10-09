

import json
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import random
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw

from tqdm import tqdm
from joblib import Parallel, delayed

import re
import ast
import dashscope
from dashscope import MultiModalConversation
import base64
import time


from prompts.target_area_semantic_accuracy import *
from prompts.whole_semantic_accuracy import *
from prompts.whole_perceptual_quality import *


def mask_img(img_path, save_path, bbox_list):
    '''
    img_path: path of the source image
    save_path: path to save to
    bbox_list: list of bboxes, format [[x1, y1, x2, y2], ...]
    Paint the bbox regions of img_path white and save to save_path.
    '''
    try:
        # 1. open the image
        # convert("RGB") avoids errors on palette or greyscale images,
        # also compatible with RGBA (transparent background becomes white, as required)
        with Image.open(img_path).convert("RGB") as img:
            
            # 2. create the drawing object
            draw = ImageDraw.Draw(img)
            
            # 3. fill every bbox in the list
            for bbox in bbox_list:
                # bbox format is usually [x1, y1, x2, y2] (top-left x, top-left y, bottom-right x, bottom-right y)
                # PIL's rectangle() accepts this format directly
                # fill="white" paints white; outline=None means no border
                draw.rectangle(bbox, fill="white", outline=None)
            
            # 4. make sure the output directory exists (optional but recommended)
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 5. save the image
            img.save(save_path)


    except Exception as e:
        print(f"error while processing the image: {e}")
        raise

def parse_evaluation_response(response_text):
    """
    Robustly parse the evaluation JSON returned by the LLM.
    
    Args:
        response_text (str): raw output string of the LLM
        
    Returns:
        dict: with 'score', 'reasoning', 'detected_text'.
              On parse failure a default error dict (score=-1) is returned.
    """
    
    # default failure return, prevents crashes
    default_result = {
        "score": -1 # -1 marks a parse failure
    }

    if not response_text:
        return default_result

    # 1. pre-process: strip Markdown code fences (```json ... ```)
    # not strictly needed (the regex below also copes) but safer
    clean_text = re.sub(r'```json\s*|\s*```', '', response_text, flags=re.IGNORECASE)

    # 2. extract the core: regex for the outermost {}
    # re.DOTALL lets . match newlines
    match = re.search(r'\{[\s\S]*\}', clean_text)
    
    if not match:
        print(f"[Warning] no JSON structure found in output: {response_text[:50]}...")
        return default_result
        
    json_str = match.group()

    parsed_data = None

    # 3. plan A: standard JSON parsing (strictest, fastest)
    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError:
        # 4. plan B: Python AST parsing (more tolerant)
        # if the model outputs {'score': 9} (single quotes) json.loads fails but ast.literal_eval works
        try:
            parsed_data = ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            # 5. plan C: last resort - json_repair (if installed)
            # avoid a third-party dependency: try a simple manual repair, or give up
            print(f"[Error] JSON parsing failed completely: \n{json_str}")
            return default_result

    # 6. validation and type conversion
    if not isinstance(parsed_data, dict):
        return default_result

    # make sure score is an integer
    try:
        raw_score = parsed_data.get("score", -1)
        parsed_data["score"] = int(raw_score)
    except (ValueError, TypeError):
        parsed_data["score"] = -1 # treat non-integer scores as failures


    return parsed_data

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def crop_pad_and_resize(image_path, output_path, bbox, target_size=(512, 512)):
    """
    Crop the bbox region, pad to a square with white, and resize to the target size.
    
    Args:
        image_path (str): path of the source image
        output_path (str): path to save to
        bbox (list or tuple): crop region [x1, y1, x2, y2]
        target_size (tuple): final output size, default (512, 512)
    """

    # 1. open the image
    img = Image.open(image_path)
    
    # convert the colour mode so RGBA can be saved as JPG without black backgrounds
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 2. crop the image (x1, y1, x2, y2)
    # make sure the coordinates are integers
    bbox = list(map(int, bbox))
    cropped_img = img.crop(bbox)
    
    # read the cropped size
    w, h = cropped_img.size
    
    # 3. pad to a square (white background)
    # the long side becomes the square side
    max_side = max(w, h)
    
    # create a white square background (255, 255, 255)
    square_img = Image.new('RGB', (max_side, max_side), (255, 255, 255))
    
    # compute the paste offset so the crop is centred
    # (long side - short side) // 2
    offset_x = (max_side - w) // 2
    offset_y = (max_side - h) // 2
    
    # paste the crop onto the white background
    square_img.paste(cropped_img, (offset_x, offset_y))
    
    # 4. resize proportionally to 512x512
    # LANCZOS filter for high-quality rescaling
    final_img = square_img.resize(target_size, Image.Resampling.LANCZOS)
    
    # 5. save the image
    # make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_img.save(output_path)



def get_vlm_result(message_list):
    messages = [
        {
            "role": "user",
            "content": message_list
        }
    ]

    response = MultiModalConversation.call(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model="qwen3-vl-flash",
        messages=messages,
        stream=False,
        enable_thinking=True, # TODO
        thinking_budget=81920,
    )
    try:
        text = response['output']['choices'][0]['message']['content'][0]['text']
        results = parse_evaluation_response(text)
        # print(results)
        return results
    except Exception as e:
        print(response)
        print(e)
        for i in range(5):
            time.sleep(60)
            print(f'Retrying... {i+1}')
            response = MultiModalConversation.call(
                api_key=os.environ["DASHSCOPE_API_KEY"],
                model="qwen3-vl-flash",
                messages=messages,
                stream=False,
                enable_thinking=True,
                thinking_budget=81920,
            )
            try:
                text = response['output']['choices'][0]['message']['content'][0]['text']
                results = parse_evaluation_response(text)
                return results
            except Exception as e:
                continue
        default_result = {
            "score": -1 # -1 marks a parse failure
        }
        return default_result




def target_area_exec_task(func_input):
    '''
    how the function input is processed into the function output
    {
        'label':
        'img_path'
        'prompt'
        'edit_subtype'
        'edit_prompt'
    }

    or

    {
        'label':
        'img_path_old' The Reference Image
        'img_path_new' The Edited Image
        'prompt'
        'edit_subtype'
        'edit_prompt'
    }

    '''
    def get_img_url(image_path):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        base64_image = encode_image(image_path)
        image_type = image_path.split(".")[-1]
        if image_type == "png":
            img_url = f"data:png;base64,{base64_image}"
        elif image_type in ["jpg", "jpeg"]:
            img_url = f"data:jpeg;base64,{base64_image}"
        else:
            raise

        return img_url

    if 'img_path' in func_input:
        img_url = get_img_url(func_input['img_path'])
        messages = [
            {"text": func_input['prompt']},
            {"image": img_url}
        ]
    else:
        img_url1 = get_img_url(func_input['img_path_old'])
        img_url2 = get_img_url(func_input['img_path_new'])
        messages = [
            {"text": func_input['prompt']},
            {"text": "The Reference Image"},
            {"image": img_url1},
            {"text": "The Edited Image"},
            {"image": img_url2}
        ]

    func_output = get_vlm_result(messages)
    func_input['gpt_judgement'] = func_output
    return func_input



def whole_exec_sc_task(func_input):
    '''
    how the function input is processed into the function output
    {
        'label':
        'img_path_old'
        'img_path_new'
        'prompt'
        'edit_subtype'
        'edit_prompt'
    }

    '''
    def get_img_url(image_path):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        base64_image = encode_image(image_path)
        image_type = image_path.split(".")[-1]
        if image_type == "png":
            img_url = f"data:png;base64,{base64_image}"
        elif image_type in ["jpg", "jpeg"]:
            img_url = f"data:jpeg;base64,{base64_image}"
        else:
            raise

        return img_url


    img_url1 = get_img_url(func_input['img_path_old'])
    img_url2 = get_img_url(func_input['img_path_new'])
    messages = [
        {"text": func_input['prompt']},
        {"text": "The Original Image"},
        {"image": img_url1},
        {"text": "The Edited Image"},
        {"image": img_url2}
    ]

    func_output = get_vlm_result(messages)
    func_input['gpt_judgement'] = func_output
    return func_input


def whole_exec_pq_task(func_input):
    '''
    how the function input is processed into the function output
    {
        'label':
        'img_path_old'
        'img_path_new'
        'prompt'
        'edit_subtype'
        'edit_prompt'
    }

    '''
    def get_img_url(image_path):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        base64_image = encode_image(image_path)
        image_type = image_path.split(".")[-1]
        if image_type == "png":
            img_url = f"data:png;base64,{base64_image}"
        elif image_type in ["jpg", "jpeg"]:
            img_url = f"data:jpeg;base64,{base64_image}"
        else:
            raise

        return img_url


    img_url1 = get_img_url(func_input['img_path_old'])
    img_url2 = get_img_url(func_input['img_path_new'])
    messages = [
        {"text": func_input['prompt']},
        {"text": "The Original Image"},
        {"image": img_url1},
        {"text": "The Compressed Image"},
        {"image": img_url2}
    ]

    func_output = get_vlm_result(messages)
    func_input['gpt_judgement'] = func_output
    return func_input


def multiprocessing_tasks(input_list, batch_size, exec_func):
    tasks = []
    for func_input in input_list:
        tasks.append(delayed(exec_func)(func_input))

    output_list = Parallel(n_jobs=batch_size, backend='multiprocessing')(
        tqdm(tasks, total=len(tasks), desc='Processing tasks')
    )

    return output_list


def judge_target_area_semantic_accuracy(data, save_dir, batch_size):
    original_imgs_root = data['original_imgs_root']
    edited_imgs_root = data['edited_imgs_root']
    data_list = data['data_list']

    input_list = []
    for key in data_list.keys():
        datas = data_list[key]
        for item in datas:
            # item[0] is the file name
            ori_path = os.path.join(original_imgs_root, key, item[0])
            edit_path = os.path.join(edited_imgs_root, key, item[0])
            edit_prompt = item[1]
            edit_caption = item[2]
            edit_subtype = item[3]
            edit_target = item[4]
            bbox_list = item[5] # keep the index alignment correct
            
            if edit_subtype in ['add']:
                params = {}
                cropped_save_path = os.path.join(save_dir, edit_subtype, item[0])
                os.makedirs(os.path.dirname(cropped_save_path), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path, bbox_list[0])

                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path'] = cropped_save_path
                params['prompt'] = tasc_add.format(text=edit_target)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt
                input_list.append(params)
            
            elif edit_subtype in ['color', 'gradient', 'texture']:
                params = {}
                cropped_save_path = os.path.join(save_dir, edit_subtype, item[0])
                os.makedirs(os.path.dirname(cropped_save_path), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path, bbox_list[0])

                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path'] = cropped_save_path
                params['prompt'] = tasc_color_gradient_texture.format(text=edit_target, statement=edit_caption)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt
                input_list.append(params)
            
            elif edit_subtype in ['font']:
                params = {}
                cropped_save_path = os.path.join(save_dir, edit_subtype, item[0])
                os.makedirs(os.path.dirname(cropped_save_path), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path, bbox_list[0])

                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path'] = cropped_save_path
                params['prompt'] = tasc_font.format(text=edit_target, statement=edit_caption)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)
            
            elif edit_subtype in ['correct']:
                params = {}
                cropped_save_path = os.path.join(save_dir, edit_subtype, item[0])
                os.makedirs(os.path.dirname(cropped_save_path), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path, bbox_list[0])

                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path'] = cropped_save_path
                params['prompt'] = tasc_correct.format(text=edit_target.split('-')[1])
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)

            elif edit_subtype in ['edit']:
                params = {}
                cropped_save_path = os.path.join(save_dir, edit_subtype, item[0])
                os.makedirs(os.path.dirname(cropped_save_path), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path, bbox_list[0])

                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path'] = cropped_save_path
                params['prompt'] = tasc_edit.format(text_1=edit_target.split('-')[0], text_2=edit_target.split('-')[1])
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)

            elif edit_subtype in ['weight', 'size']:
                params = {}
                cropped_save_path_old = os.path.join(save_dir, edit_subtype+'_old', item[0])
                os.makedirs(os.path.dirname(cropped_save_path_old), exist_ok=True)
                crop_pad_and_resize(ori_path, cropped_save_path_old, bbox_list[0])

                cropped_save_path_new = os.path.join(save_dir, edit_subtype+'_new', item[0])
                os.makedirs(os.path.dirname(cropped_save_path_new), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path_new, bbox_list[0])

                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = cropped_save_path_old
                params['img_path_new'] = cropped_save_path_new
                params['prompt'] = tasc_weight_size.format(text=edit_target, instruction=edit_prompt)

                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt
                input_list.append(params)
            
            elif edit_subtype in ['remove']:
                params = {}
                cropped_save_path = os.path.join(save_dir, edit_subtype, item[0])
                os.makedirs(os.path.dirname(cropped_save_path), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path, bbox_list[0])

                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path'] = cropped_save_path
                params['prompt'] = tasc_remove.format(text_to_remove=edit_target)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)

            elif edit_subtype in ['rotate']:
                params = {}
                cropped_save_path = os.path.join(save_dir, edit_subtype, item[0])
                os.makedirs(os.path.dirname(cropped_save_path), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path, bbox_list[0])

                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path'] = cropped_save_path
                params['prompt'] = tasc_rotate.format(text=edit_target, description=edit_caption)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)
            
            elif edit_subtype in ['move']:
                params_remove = {}
                cropped_save_path_remove = os.path.join(save_dir, edit_subtype + "_remove", item[0])
                os.makedirs(os.path.dirname(cropped_save_path_remove), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path_remove, bbox_list[0])

                params_remove['label'] = edit_subtype + '_remove\\' + item[0]
                params_remove['img_path'] = cropped_save_path_remove
                params_remove['prompt'] = tasc_remove.format(text_to_remove=edit_target)
                params_remove['edit_subtype'] = edit_subtype + '_remove'
                params_remove['edit_prompt'] = edit_prompt                
                input_list.append(params_remove)

                params_add = {}
                cropped_save_path_add = os.path.join(save_dir, edit_subtype + "_add", item[0])
                os.makedirs(os.path.dirname(cropped_save_path_add), exist_ok=True)
                crop_pad_and_resize(edit_path, cropped_save_path_add, bbox_list[1])

                params_add['label'] = edit_subtype + '_add\\' + item[0]
                params_add['img_path'] = cropped_save_path_add
                params_add['prompt'] = tasc_add.format(text=edit_target)
                params_add['edit_subtype'] = edit_subtype + '_add'
                params_add['edit_prompt'] = edit_prompt                
                input_list.append(params_add)
            else:
                raise

    output_list = multiprocessing_tasks(input_list, batch_size, target_area_exec_task)
    return output_list


def judge_target_area_perceptual_quality(data, save_dir, batch_size):
    '''
    not needed
    '''
    pass

def judge_whole_semantic_accuracy(data, save_dir, batch_size):
    original_imgs_root = data['original_imgs_root']
    edited_imgs_root = data['edited_imgs_root']
    data_list = data['data_list']

    input_list = []
    for key in data_list.keys():
        datas = data_list[key]
        for item in datas:
            # item[0] is the file name
            ori_path = os.path.join(original_imgs_root, key, item[0])
            edit_path = os.path.join(edited_imgs_root, key, item[0])
            edit_prompt = item[1]
            edit_caption = item[2]
            edit_subtype = item[3]
            edit_target = item[4]
            bbox_list = item[5] # keep the index alignment correct
            
            if edit_subtype in ['add']:
                params = {}
                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = ori_path
                params['img_path_new'] = edit_path
                params['prompt'] = wsc_add.format(text=edit_target, instruction=edit_prompt)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt
                input_list.append(params)
            
            elif edit_subtype in ['color', 'gradient', 'texture']:
                params = {}
                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = ori_path
                params['img_path_new'] = edit_path
                params['prompt'] = wsc_color_gradient_texture.format(text=edit_target, instruction=edit_prompt)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt
                input_list.append(params)
            
            elif edit_subtype in ['font']:
                params = {}
                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = ori_path
                params['img_path_new'] = edit_path
                params['prompt'] = wsc_font.format(text=edit_target, instruction=edit_prompt)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)
            
            elif edit_subtype in ['correct']:
                params = {}
                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = ori_path
                params['img_path_new'] = edit_path
                params['prompt'] = wsc_correct.format(instruction=edit_prompt, original_text=edit_target.split('-')[0], target_text=edit_target.split('-')[1])
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)

            elif edit_subtype in ['edit']:
                params = {}
                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = ori_path
                params['img_path_new'] = edit_path
                params['prompt'] = wsc_edit.format(text_1=edit_target.split('-')[0], text_2=edit_target.split('-')[1], instruction=edit_prompt)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)

            elif edit_subtype in ['weight', 'size']:
                params = {}
                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = ori_path
                params['img_path_new'] = edit_path
                params['prompt'] = wsc_weight_size.format(text=edit_target, instruction=edit_prompt)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt
                input_list.append(params)
            
            elif edit_subtype in ['remove']:
                params = {}
                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = ori_path
                params['img_path_new'] = edit_path
                params['prompt'] = wsc_remove.format(text_to_remove=edit_target, instruction=edit_prompt)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)

            elif edit_subtype in ['rotate']:
                params = {}
                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = ori_path
                params['img_path_new'] = edit_path
                params['prompt'] = wsc_rotate.format(text=edit_target, instruction=edit_prompt)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)
            
            elif edit_subtype in ['move']:
                params = {}
                params['label'] = edit_subtype + '\\' + item[0]
                params['img_path_old'] = ori_path
                params['img_path_new'] = edit_path
                params['prompt'] = wsc_move.format(text_to_move=edit_target, editing_instruction=edit_prompt)
                params['edit_subtype'] = edit_subtype
                params['edit_prompt'] = edit_prompt                
                input_list.append(params)

            else:
                raise

    output_list = multiprocessing_tasks(input_list, batch_size, whole_exec_sc_task)
    return output_list


def judge_whole_perceptual_quality(data, save_dir, batch_size):
    original_imgs_root = data['original_imgs_root']
    edited_imgs_root = data['edited_imgs_root']
    data_list = data['data_list']

    input_list = []
    for key in data_list.keys():
        datas = data_list[key]
        for item in datas:
            # item[0] is the file name
            ori_path = os.path.join(original_imgs_root, key, item[0])
            edit_path = os.path.join(edited_imgs_root, key, item[0])
            edit_prompt = item[1]
            edit_caption = item[2]
            edit_subtype = item[3]
            edit_target = item[4]
            bbox_list = item[5] # keep the index alignment correct
            
            tmp_ori_path = os.path.join(save_dir, 'original', edit_subtype, item[0])
            tmp_edit_path = os.path.join(save_dir, 'edited', edit_subtype, item[0])

            os.makedirs(os.path.dirname(tmp_ori_path), exist_ok=True)
            os.makedirs(os.path.dirname(tmp_edit_path), exist_ok=True)

            mask_img(ori_path, tmp_ori_path, bbox_list)
            mask_img(edit_path, tmp_edit_path, bbox_list)

            params = {}
            params['label'] = edit_subtype + '\\' + item[0]
            params['img_path_old'] = tmp_ori_path
            params['img_path_new'] = tmp_edit_path
            params['prompt'] = wpq
            params['edit_subtype'] = edit_subtype
            params['edit_prompt'] = edit_prompt                
            input_list.append(params)

    output_list = multiprocessing_tasks(input_list, batch_size, whole_exec_pq_task)
    return output_list


def run_evaluation(data_path, save_root, batch_size):
    os.makedirs(save_root, exist_ok=True)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for _k in ("original_imgs_root", "edited_imgs_root"):
            if isinstance(data.get(_k), str) and not os.path.isabs(data[_k]):
                data[_k] = os.path.join(REPO_ROOT, data[_k])


    print('start calculating target area semantic accuracy')
    tmp_save_dir = os.path.join(save_root, 'tmp_tasa_dir')
    output = judge_target_area_semantic_accuracy(data, tmp_save_dir, batch_size)
    result_path = os.path.join(save_root, 'gpt_target_area_semantic_accuracy.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    # print('start calculating target area perceptual quality')
    # tmp_save_dir = os.path.join(save_root, 'tmp_tasa_dir')
    # output = judge_target_area_perceptual_quality(data, tmp_save_dir, batch_size)
    # result_path = os.path.join(save_root, 'gpt_target_area_perceptual_quality.json')
    # with open(result_path, 'w', encoding='utf-8') as f:
    #     json.dump(output, f, indent=4, ensure_ascii=False)

    print('start calculating whole semantic accuracy')
    tmp_save_dir = os.path.join(save_root, 'tmp_wsc_dir')
    output = judge_whole_semantic_accuracy(data, tmp_save_dir, batch_size)
    result_path = os.path.join(save_root, 'gpt_whole_semantic_accuracy.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print('start calculating whole perceptual quality')
    tmp_save_dir = os.path.join(save_root, 'tmp_wpq_dir')
    output = judge_whole_perceptual_quality(data, tmp_save_dir, batch_size)
    result_path = os.path.join(save_root, 'gpt_whole_perceptual_quality.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default=os.path.join(REPO_ROOT, "benchmark", "validation.json"))
    parser.add_argument('--save_dir', type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    run_evaluation(args.json_path, args.save_dir, args.batch_size)
