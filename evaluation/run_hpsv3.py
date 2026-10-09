import torch
import torch.nn.functional as F
import json
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import argparse
from PIL import Image
# from transformers.image_utils import load_image # unused; commented out to reduce dependencies
from transformers import AutoModel, AutoProcessor
import random
import numpy as np
from hpsv3 import HPSv3RewardInferencer
from tqdm import tqdm  # progress bar

def seed_everything(seed=42):
    """
    Fix every possible random seed for reproducibility.
    """
    random.seed(seed)                 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)              
    torch.manual_seed(seed)           
    torch.cuda.manual_seed(seed)      
    torch.cuda.manual_seed_all(seed)  
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    [new] if the final square is smaller than 256x256, force-upscale to 256x256.
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
    side_length = max(width, height)
    
    # 3. create a white square canvas
    square_img = Image.new("RGB", (side_length, side_length), (255, 255, 255))
    
    # 4. compute the centring offset
    # offset = (long side - short side) // 2
    left = (side_length - width) // 2
    top = (side_length - height) // 2
    
    # 5. paste the source image
    square_img.paste(rgb_img, (left, top))
    
    # 6. [new] check the size and upscale if needed
    # if the longest side is below 256 the whole image is smaller than 256x256: upscale
    if side_length < 256:
        # BICUBIC interpolation keeps the upscaled image sharp
        square_img = square_img.resize((256, 256), Image.BICUBIC)
    
    return square_img



def calc_hps(img_paths, caption_list, scorer, batch_size=8):
    """
    [Critical Fix] scalar-conversion error caused by Tensor dimensionality
    """
    all_scores = []
    num_batches = (len(img_paths) + batch_size - 1) // batch_size
    print(f"computing HPS for {len(img_paths)} images in {num_batches} batches...")

    for i in tqdm(range(0, len(img_paths), batch_size), total=num_batches, desc="HPS Evaluation"):
        batch_paths = img_paths[i : i + batch_size]
        batch_captions = caption_list[i : i + batch_size]
        
        try:
            with torch.no_grad():
                rewards = scorer.reward(batch_captions, image_paths=batch_paths)
            
            # --- core fix ---
            if isinstance(rewards, torch.Tensor):
                # check dims: 2-D (Batch, 2) or (Batch, 1)
                if rewards.ndim == 2:
                    # take the first column (Score), ignore the second (std or other)
                    # .cpu().tolist() converts the Tensor straight into a float list
                    batch_scores = rewards[:, 0].cpu().tolist()
                elif rewards.ndim == 1:
                    # 1-D case (rare, but handled for robustness)
                    batch_scores = rewards.cpu().tolist()
                else:
                    # very rare higher-dimensional case: take the first element
                    batch_scores = [r.view(-1)[0].item() for r in rewards]
            
            elif isinstance(rewards, list):
                # if it is a list, handle it compatibly
                batch_scores = []
                for r in rewards:
                    if isinstance(r, torch.Tensor):
                        # if the inner tensor is also multi-dimensional, take the first element
                        batch_scores.append(r.view(-1)[0].item())
                    elif isinstance(r, (list, tuple)):
                         batch_scores.append(float(r[0]))
                    else:
                        batch_scores.append(float(r))
            else:
                batch_scores = [float(rewards)]
            
            # make sure the result is a flat list
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
                
            all_scores.extend(batch_scores)
            
        except Exception as e:
            print(f"batch {i//batch_size} failed: {e}")
            # print the rewards shape for debugging if problems persist
            try:
                print(f"Rewards shape: {rewards.shape if isinstance(rewards, torch.Tensor) else type(rewards)}")
            except:
                pass
            all_scores.extend([0.0] * len(batch_paths))

    return all_scores


def calc_target_area_hps(data_info, scorer, tmp_dir, batch_size=8):
    original_imgs_root = data_info['original_imgs_root']
    edited_imgs_root = data_info['edited_imgs_root']
    data_list = data_info['data_list']

    label_list = []
    edited_img_target_area_path = []
    caption_list = []

    for key in data_list.keys():
        for item in data_list[key]:

            edited_img_path = os.path.join(edited_imgs_root, key, item[0])
            tmp_edited_path = os.path.join(tmp_dir, key, item[0])
            os.makedirs(os.path.join(tmp_dir, key), exist_ok=True)
            prompt = item[1]
            caption = item[2]
            edit_subtype = item[3]
            edit_target = item[4]
            bbox_list = item[5]
            comment = item[6]

            if edit_subtype == 'add':
                label_list.append(edit_subtype + '\\' + item[0])
                if not os.path.exists(tmp_edited_path):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[0]))
                    cropped_image.save(tmp_edited_path)

                edited_img_target_area_path.append(tmp_edited_path)
                caption_list.append(f"Rendered text '{edit_target}' appears in the image.")
            elif edit_subtype in ['color', 'gradient', 'texture', 'font', 'weight', 'rotate']:
                label_list.append(edit_subtype + '\\' + item[0])
                if not os.path.exists(tmp_edited_path):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[0]))
                    cropped_image.save(tmp_edited_path)

                edited_img_target_area_path.append(tmp_edited_path)
                caption_list.append(caption)
            elif edit_subtype == 'size':
                continue
            elif edit_subtype == 'edit':
                label_list.append(edit_subtype + '\\' + item[0])
                if not os.path.exists(tmp_edited_path):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[0]))
                    cropped_image.save(tmp_edited_path)

                edited_img_target_area_path.append(tmp_edited_path)
                caption_list.append(f"Rendered text '{edit_target.split('-')[1]}' appears in the image.")
            elif edit_subtype == 'correct':
                label_list.append(edit_subtype + '\\' + item[0])
                if not os.path.exists(tmp_edited_path):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[0]))
                    cropped_image.save(tmp_edited_path)

                edited_img_target_area_path.append(tmp_edited_path)
                caption_list.append(f"The text '{edit_target.split('-')[1]}' is correctly rendered in the image.")
            elif edit_subtype == 'move':
                label_list.append(edit_subtype + '_old\\' + item[0])
                tmp_edited_path_old = tmp_edited_path.replace(key, key+'_old')
                os.makedirs(os.path.dirname(tmp_edited_path_old), exist_ok=True)
                if not os.path.exists(tmp_edited_path_old):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[0]))
                    cropped_image.save(tmp_edited_path_old)
                edited_img_target_area_path.append(tmp_edited_path_old)
                caption_list.append(f"There is no text '{edit_target}' in the image")

                label_list.append(edit_subtype + '_new\\' + item[0])
                tmp_edited_path_new = tmp_edited_path.replace(key, key+'_new')
                os.makedirs(os.path.dirname(tmp_edited_path_new), exist_ok=True)
                if not os.path.exists(tmp_edited_path_new):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[1]))
                    cropped_image.save(tmp_edited_path_new)
                edited_img_target_area_path.append(tmp_edited_path_new)
                caption_list.append(f"Rendered text '{edit_target}' appears in the image.")
            elif edit_subtype == 'remove':
                label_list.append(edit_subtype + '\\' + item[0])
                if not os.path.exists(tmp_edited_path):
                    cropped_image = pad_image_to_square(crop_image(edited_img_path, bbox_list[0]))
                    cropped_image.save(tmp_edited_path)

                edited_img_target_area_path.append(tmp_edited_path)
                caption_list.append(f"There is no text '{edit_target}' in the image")
            else:
                raise

    # [bug fix] unpacking a single variable here used to raise
    # calc_hps now returns only a list
    per_sample_scores = calc_hps(edited_img_target_area_path, caption_list, scorer, batch_size=batch_size)
    
    # compute the total (mean) score
    if len(per_sample_scores) > 0:
        total_score = sum(per_sample_scores) / len(per_sample_scores)
    else:
        total_score = 0.0

    # per_sample_scores is already list[float]; no .cpu().tolist() needed
    return {
        "total_score": total_score, 
        "per_sample_scores": per_sample_scores, 
        "label_list": label_list, 
        'caption_list': caption_list
    }




def calc_whole_hps(data_info, scorer, batch_size=8):
    original_imgs_root = data_info['original_imgs_root']
    edited_imgs_root = data_info['edited_imgs_root']
    data_list = data_info['data_list']

    label_list = []
    edited_img_path_list = []
    caption_list = []

    for key in data_list.keys():
        for item in data_list[key]:

            edited_img_path = os.path.join(edited_imgs_root, key, item[0])
            prompt = item[1]
            caption = item[2]
            edit_subtype = item[3]
            edit_target = item[4]
            bbox_list = item[5]
            comment = item[6]

            label_list.append(edit_subtype + '\\' + item[0])
            edited_img_path_list.append(edited_img_path)
            caption_list.append(caption)


    # [bug fix] unpacking a single variable here used to raise
    # calc_hps now returns only a list
    per_sample_scores = calc_hps(edited_img_path_list, caption_list, scorer, batch_size=batch_size)
    
    # compute the total (mean) score
    if len(per_sample_scores) > 0:
        total_score = sum(per_sample_scores) / len(per_sample_scores)
    else:
        total_score = 0.0

    # per_sample_scores is already list[float]; no .cpu().tolist() needed
    return {
        "total_score": total_score, 
        "per_sample_scores": per_sample_scores, 
        "label_list": label_list, 
        'caption_list': caption_list
    }





def run_evaluation(data, scorer, save_dir, batch_size):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tmp_dir = os.path.join(save_dir, 'hpsv3_tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # cropping the region is not appropriate for this metric
    # print(f"Processing Target Area hps (Batch Size: {batch_size})...")
    # output = calc_target_area_hps(data, scorer, tmp_dir, batch_size=batch_size)
    
    # save_path = os.path.join(save_dir, "hpsv3_target_area.json")
    # with open(save_path, 'w', encoding='utf-8') as f:
    #     json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Processing whole hps (Batch Size: {batch_size})...")
    output = calc_whole_hps(data, scorer, batch_size=batch_size)
    
    save_path = os.path.join(save_dir, "hpsv3_whole.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    seed_everything(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default=os.path.join(REPO_ROOT, "benchmark", "validation.json"), help="path of the benchmark json")
    parser.add_argument('--model_path', type=str, default=os.path.join(REPO_ROOT, "weights", "MizzenAI__HPSv3", "HPSv3.safetensors"))
    parser.add_argument('--save_dir', type=str, default=os.path.join(REPO_ROOT, "results"))
    # batch_size argument, default 8; lower to 1 or 2 on small GPUs
    parser.add_argument('--batch_size', type=int, default=4, help="inference batch size; lower it if VRAM is tight")
    
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        exit()

    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for _k in ("original_imgs_root", "edited_imgs_root"):
            if isinstance(data.get(_k), str) and not os.path.isabs(data[_k]):
                data[_k] = os.path.join(REPO_ROOT, data[_k])

    print("Loading Model...")
    model_path = args.model_path
    # make sure the device is correct
    scorer = HPSv3RewardInferencer(checkpoint_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = args.save_dir
    
    try:
        run_evaluation(data, scorer, save_dir, args.batch_size)
        print("all evaluations finished and saved successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"runtime error: {e}")