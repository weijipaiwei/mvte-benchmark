import json
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import random
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
import torchmetrics.functional as F
import torchvision.transforms as T
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Initializing metrics on {self.device}...")
        
        # data_range=1.0 everywhere because the inputs are floats in [0, 1]
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # vgg is recommended for LPIPS
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device)
        # FID needs a feature extractor
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        
        self.lpips.eval()

    @torch.no_grad()
    def update_batch(self, real_imgs, fake_imgs):
        """
        input: [B, C, H, W], float32, range [0, 1]
        """
        real_imgs = real_imgs.to(self.device)
        fake_imgs = fake_imgs.to(self.device)

        # --- 1. FID ---
        # normalize=True converts floats in [0, 1] to uint8 inside TorchMetrics.
        # Pass floats directly so the images are scaled exactly once.
        self.fid.update(real_imgs, real=True)
        self.fid.update(fake_imgs, real=False)
        
        # --- 2. paired metrics (vectorised) ---
        batch_size = real_imgs.shape[0]

        # [PSNR] 
        # dim=(1,2,3) is required to aggregate over the non-batch dims
        psnr_batch = F.peak_signal_noise_ratio(
            fake_imgs, real_imgs, 
            data_range=1.0, 
            reduction='none', 
            dim=(1, 2, 3) 
        )
        psnr_batch = torch.nan_to_num(psnr_batch, posinf=100.0)
        
        # [SSIM] 
        # reduction='none' returns [B]
        ssim_batch = F.structural_similarity_index_measure(
            fake_imgs, real_imgs, 
            data_range=1.0, 
            reduction='none'
        )

        # [LPIPS]
        # normalize=True maps [0,1] -> [-1,1]
        fake_scaled = fake_imgs * 2 - 1
        real_scaled = real_imgs * 2 - 1
        # the net returns [B, 1, 1, 1]; it must be flattened
        lpips_batch = self.lpips.net(fake_scaled, real_scaled)

        # --- 3. return the results ---
        # .view(-1) is the key:
        # 1. fix PSNR/SSIM degenerating to a scalar when Batch=1
        # 2. fix the redundant [B,1,1,1] dims of LPIPS
        
        return {
            # running sum (for the global average)
            "PSNR_SUM": psnr_batch.sum().item(),
            "SSIM_SUM": ssim_batch.sum().item(),
            "LPIPS_SUM": lpips_batch.sum().item(),
            
            # list (keeps the per-sample details)
            "PSNR_list": psnr_batch.view(-1).cpu().tolist(),
            "SSIM_list": ssim_batch.view(-1).cpu().tolist(),
            "LPIPS_list": lpips_batch.view(-1).cpu().tolist(),
            
            "COUNT": batch_size
        }

    def compute_fid(self):
        # FID may warn (or fail) with too few samples (warning below 2048, but still computable)
        try:
            final_fid = self.fid.compute().item()
        except Exception as e:
            print(f"Warning: FID computation failed (possibly too few samples): {e}")
            final_fid = -1.0
        self.fid.reset()
        return final_fid

def process_single_image(path, bbox_list, target_size=512, mask_bbox=False):
    """
    Helper: process a single image.
    mask_bbox: whether to paint the bbox region white
    """
    img = Image.open(path).convert('RGB')
    
    # --- control: whether to paint the bbox white ---
    if mask_bbox and bbox_list is not None:
        draw = ImageDraw.Draw(img)
        for bbox in bbox_list:
            draw.rectangle(tuple(bbox), fill=(255, 255, 255))
    # ----------------------------

    w, h = img.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.BICUBIC)

    new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    new_img.paste(img, (pad_w, pad_h))
    
    return new_img

def load_paired_batch(batch_pairs, target_size=512, mask_bbox=False):
    """
    [fix] load image pairs together to keep them aligned
    """
    to_tensor = T.ToTensor()
    ori_tensors = []
    edit_tensors = []
    batch_label_list = []
    
    for ori_path, edit_path, bbox_list, label in batch_pairs:
        try:
            # process both images together
            ori_pil = process_single_image(ori_path, bbox_list, target_size, mask_bbox)
            edit_pil = process_single_image(edit_path, bbox_list, target_size, mask_bbox)
            
            # append only if both images succeeded
            ori_tensors.append(to_tensor(ori_pil))
            edit_tensors.append(to_tensor(edit_pil))
            batch_label_list.append(label)
            
        except Exception as e:
            # print a short error without interrupting the run
            print(f"[Warn] Skip pair due to error: {os.path.basename(ori_path)} - {e}")
            continue

    if not ori_tensors:
        return None, None
        
    return torch.stack(ori_tensors), torch.stack(edit_tensors), batch_label_list

def run_evaluation(data, save_dir, batch_size=32, mask_bbox=True):
    os.makedirs(save_dir, exist_ok=True)
    evaluator = ImageEvaluator()
    
    original_imgs_root = data['original_imgs_root']
    edited_imgs_root = data['edited_imgs_root']
    data_list = data['data_list']

    pairs = []
    for key in data_list.keys():
        datas = data_list[key]
        for item in datas:
            # item[0] is the file name
            ori_path = os.path.join(original_imgs_root, key, item[0])
            edit_path = os.path.join(edited_imgs_root, key, item[0])
            edit_subtype = item[3]
            bbox_list = item[5] # keep the index alignment correct
            pairs.append((ori_path, edit_path, bbox_list, edit_subtype+'\\'+item[0]))
    
    total_samples = len(pairs)
    print(f"Total pairs found: {total_samples}")

    metrics_sum = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0, "PSNR_list": [], "SSIM_list": [], "LPIPS_list": [], "label_list": []}
    valid_samples_count = 0

    # show progress with tqdm
    for i in tqdm(range(0, total_samples, batch_size), desc="Evaluating"):
        batch_pairs = pairs[i : i + batch_size]
        
        # [fix] load in pairs
        ori_imgs, edit_imgs, batch_label_list = load_paired_batch(batch_pairs, target_size=512, mask_bbox=mask_bbox)
        
        if ori_imgs is None:
            continue
            
        # run the evaluation
        batch_res = evaluator.update_batch(ori_imgs, edit_imgs)
        
        metrics_sum["PSNR"] += batch_res["PSNR_SUM"]
        metrics_sum["SSIM"] += batch_res["SSIM_SUM"]
        metrics_sum["LPIPS"] += batch_res["LPIPS_SUM"]
        metrics_sum["PSNR_list"] += batch_res["PSNR_list"]
        metrics_sum["SSIM_list"] += batch_res["SSIM_list"]
        metrics_sum["LPIPS_list"] += batch_res["LPIPS_list"]
        metrics_sum["label_list"] += batch_label_list
        valid_samples_count += batch_res["COUNT"]


    if valid_samples_count > 0:
        metrics_sum["PSNR"] /= valid_samples_count
        metrics_sum["SSIM"] /= valid_samples_count
        metrics_sum["LPIPS"] /= valid_samples_count
    else:
        print("Error: No valid samples processed!")
        return

    print("Computing FID... (This implies loading InceptionV3, might take a moment)")
    metrics_sum["FID"] = evaluator.compute_fid()

    print("\n" + "="*30)
    print(f"Final Results ({valid_samples_count} images):")
    print('PSNR: ', metrics_sum["PSNR"])
    print('SSIM: ', metrics_sum["SSIM"])
    print('LPIPS: ', metrics_sum["LPIPS"])
    print('FID: ', metrics_sum["FID"])
    print("="*30)

    result_path = os.path.join(save_dir, "traditional_metrics.json")
    with open(result_path, 'w') as f:
        json.dump(metrics_sum, f, indent=4)
    print(f"Saved to {result_path}")

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default=os.path.join(REPO_ROOT, "benchmark", "validation.json"))
    parser.add_argument('--save_dir', type=str, default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        exit()

    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for _k in ("original_imgs_root", "edited_imgs_root"):
            if isinstance(data.get(_k), str) and not os.path.isabs(data[_k]):
                data[_k] = os.path.join(REPO_ROOT, data[_k])

    run_evaluation(data, args.save_dir, args.batch_size)
