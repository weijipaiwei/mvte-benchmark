import torch
import torch.nn.functional as F
import json
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import argparse
from PIL import Image
from transformers.image_utils import load_image
from transformers import AutoModel, AutoProcessor
import random
import numpy as np


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


class SigLIPScorerGPU:
    def __init__(self, model_name="google/siglip2-so400m-patch16-naflex", device=None, batch_size=64):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"loading model {model_name} on {self.device} (bfloat16)...")
        
        self.model = AutoModel.from_pretrained(
            model_name, 
            device_map=self.device,
            torch_dtype=torch.bfloat16 
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = batch_size
        print("model loaded.")

    def extract_image_features(self, images):
        """Extract image features in batches to avoid OOM."""
        if isinstance(images, torch.Tensor):
            return images

        all_features = []
        
        # process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            pil_images = []
            
            for img in batch_images:
                if isinstance(img, str):
                    try:
                        pil_images.append(Image.open(img).convert("RGB"))
                    except Exception as e:
                        print(f"failed to read image {img}: {e}")
                        # fill with a black image to avoid a crash, or skip if preferred
                        pil_images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
                else:
                    pil_images.append(img)
            
            if not pil_images:
                continue

            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                all_features.append(features)

        if not all_features:
            return torch.tensor([]).to(self.device)
            
        return torch.cat(all_features, dim=0)

    def extract_text_features(self, texts):
        """Extract text features in batches."""
        if isinstance(texts, torch.Tensor):
            return texts
            
        all_features = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=64) # SigLIP rarely needs 1024 tokens; a shorter length saves VRAM
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                all_features.append(features)
                
        if not all_features:
            return torch.tensor([]).to(self.device)

        return torch.cat(all_features, dim=0)

    def calculate_score(self, images, candidates, w=2.5):
        """Compute image-text matching scores."""
        img_feat = self.extract_image_features(images)
        txt_feat = self.extract_text_features(candidates)

        img_feat = F.normalize(img_feat.float(), p=2, dim=1)
        txt_feat = F.normalize(txt_feat.float(), p=2, dim=1)

        raw_scores = (img_feat * txt_feat).sum(dim=1)
        per_sample_scores = w * torch.clamp(raw_scores, min=0)

        # fix: no need to convert to list before returning; do it here or once in main.
        # keep Tensors for later computation, but convert before writing json.
        return per_sample_scores.mean().item(), per_sample_scores

    def calculate_img2img_score(self, images_a, images_b):
        img_feat_a = self.extract_image_features(images_a)
        img_feat_b = self.extract_image_features(images_b)

        if img_feat_a.shape[0] != img_feat_b.shape[0]:
            raise ValueError(f"image count mismatch: {img_feat_a.shape[0]} vs {img_feat_b.shape[0]}")

        img_feat_a = F.normalize(img_feat_a.float(), p=2, dim=1)
        img_feat_b = F.normalize(img_feat_b.float(), p=2, dim=1)

        scores = (img_feat_a * img_feat_b).sum(dim=1)
        return scores.mean().item(), scores


def calculate_img_text_score(img_path_list, prompt_list, scorer):
    total_score, per_sample_scores = scorer.calculate_score(img_path_list, prompt_list)
    return total_score, per_sample_scores

def calculate_img_img_score(img_path_a_list, img_path_b_list, scorer):
    total_score, per_sample_scores = scorer.calculate_img2img_score(img_path_a_list, img_path_b_list)
    return total_score, per_sample_scores

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
    if side_length < 512:
        # BICUBIC interpolation keeps the upscaled image sharp
        square_img = square_img.resize((256, 256), Image.BICUBIC)
    
    return square_img



def calc_target_area_img_txt_semantic(data_info, scorer):
    original_imgs_root = data_info['original_imgs_root']
    edited_imgs_root = data_info['edited_imgs_root']
    data_list = data_info['data_list']

    label_list = []
    edited_img_target_area = []
    caption_list = []

    for key in data_list.keys():
        for item in data_list[key]:
            original_img_path = os.path.join(original_imgs_root, key, item[0])
            edited_img_path = os.path.join(edited_imgs_root, key, item[0])
            prompt = item[1]
            caption = item[2]
            edit_subtype = item[3]
            edit_target = item[4]
            bbox_list = item[5]
            comment = item[6]

            if edit_subtype == 'add':
                label_list.append(edit_subtype + '\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[0])))
                caption_list.append(f"There is a text '{edit_target}' in the image")
            elif edit_subtype in ['color', 'gradient', 'texture', 'font', 'weight', 'rotate']:
                label_list.append(edit_subtype + '\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[0])))
                caption_list.append(caption)
            elif edit_subtype == 'size':
                continue
            elif edit_subtype == 'edit':
                label_list.append(edit_subtype + '_old\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[0])))
                caption_list.append(f"There is no text '{edit_target.split('-')[0]}' in the image")

                label_list.append(edit_subtype + '_new\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[0])))
                caption_list.append(f"There is a text '{edit_target.split('-')[1]}' in the image")
            elif edit_subtype == 'correct':
                label_list.append(edit_subtype + '\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[0])))
                caption_list.append(f"The text '{edit_target.split('-')[1]}' in the image is accurately spelled.")
            elif edit_subtype == 'move':
                label_list.append(edit_subtype + '_old\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[0])))
                caption_list.append(f"There is no text '{edit_target}' in the image")

                label_list.append(edit_subtype + '_new\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[1])))
                caption_list.append(f"There is a text '{edit_target}' in the image")
            elif edit_subtype == 'remove':
                label_list.append(edit_subtype + '\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[0])))
                caption_list.append(f"There is no text '{edit_target}' in the image")
            else:
                raise


    total_score, per_sample_scores = calculate_img_text_score(edited_img_target_area, caption_list, scorer)
    return {"total_score": total_score, "per_sample_scores": per_sample_scores.cpu().tolist(), "label_list": label_list, 'caption_list': caption_list}


def calc_target_area_img_img_semantic(data_info, scorer):
    gt_edited_imgs_root = data_info['ground_truth_edited_imgs_root']
    edited_imgs_root = data_info['edited_imgs_root']
    data_list = data_info['data_list']

    label_list = []
    gt_edited_img_target_area = []
    edited_img_target_area = []

    for key in data_list.keys():
        for item in data_list[key]:
            gt_edited_img_path = os.path.join(gt_edited_imgs_root, key, item[0])
            edited_img_path = os.path.join(edited_imgs_root, key, item[0])
            prompt = item[1]
            caption = item[2]
            edit_subtype = item[3]
            edit_target = item[4]
            bbox_list = item[5]
            comment = item[6]

            if edit_subtype in ['add', 'color', 'gradient', 'texture', 'font', 'weight', 'rotate', 'edit',  'correct', 'size', 'remove']:
                label_list.append(edit_subtype + '\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[0])))
                gt_edited_img_target_area.append(pad_image_to_square(crop_image(gt_edited_img_path, bbox_list[0])))
            elif edit_subtype == 'move':
                label_list.append(edit_subtype + '_old\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[0])))
                gt_edited_img_target_area.append(pad_image_to_square(crop_image(gt_edited_img_path, bbox_list[0])))

                label_list.append(edit_subtype + '_new\\' + item[0])
                edited_img_target_area.append(pad_image_to_square(crop_image(edited_img_path, bbox_list[1])))
                gt_edited_img_target_area.append(pad_image_to_square(crop_image(gt_edited_img_path, bbox_list[1])))
            else:
                raise

    total_score, per_sample_scores = calculate_img_img_score(edited_img_target_area, gt_edited_img_target_area, scorer)
    return {"total_score": total_score, "per_sample_scores": per_sample_scores.cpu().tolist(), "label_list": label_list}


def calc_edited_image_txt_semantic(data_info, scorer):
    original_imgs_root = data_info['original_imgs_root']
    edited_imgs_root = data_info['edited_imgs_root']
    data_list = data_info['data_list']

    label_list = []
    edited_img = []
    caption_list = []

    for key in data_list.keys():
        for item in data_list[key]:
            original_img_path = os.path.join(original_imgs_root, key, item[0])
            edited_img_path = os.path.join(edited_imgs_root, key, item[0])
            prompt = item[1]
            caption = item[2]
            edit_subtype = item[3]
            edit_target = item[4]
            bbox_list = item[5]
            comment = item[6]

            label_list.append(edit_subtype + '\\' + item[0])
            edited_img.append(edited_img_path)
            caption_list.append(caption)



    total_score, per_sample_scores = calculate_img_text_score(edited_img, caption_list, scorer)
    return {"total_score": total_score, "per_sample_scores": per_sample_scores.cpu().tolist(), "label_list": label_list, 'caption_list': caption_list}



def run_evaluation(data, scorer, save_dir):
    """
    Wrap the execution logic and centralise the Tensor conversion on save.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Target Area Image-Text
    print("Processing Target Area Image-Text...")
    output = calc_target_area_img_txt_semantic(data, scorer)
    with open(os.path.join(save_dir, "siglip2_target_area_img_txt.json"), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    # 2. Target Area Image-Image
    print("Processing Target Area Image-Image...")
    output = calc_target_area_img_img_semantic(data, scorer)
    with open(os.path.join(save_dir, "siglip2_target_area_img_img.json"), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    # 3. Whole Image Image-Text
    print("Processing Whole Image Image-Text...")
    output = calc_edited_image_txt_semantic(data, scorer)
    with open(os.path.join(save_dir, "siglip2_whole_image_img_txt.json"), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    seed_everything(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default=os.path.join(REPO_ROOT, "benchmark", "validation.json"), help="path of the benchmark json")
    parser.add_argument('--model_path', type=str, default=os.path.join(REPO_ROOT, "weights", "google__siglip2-so400m-patch16-naflex"))
    parser.add_argument('--save_dir', type=str, default=os.path.join(REPO_ROOT, "results"))
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

    # make sure the model path is correct
    model_path = args.model_path
    scorer = SigLIPScorerGPU(model_name=model_path, device="cuda", batch_size=64)
    save_dir = args.save_dir
    
    try:
        run_evaluation(data, scorer, save_dir)
        print("all evaluations finished and saved successfully.")
    except Exception as e:
        print(f"runtime error: {e}")