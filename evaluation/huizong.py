import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import argparse



def calc_ocr(ocr_json_path):
    with open(ocr_json_path, 'r', encoding='utf-8') as f:
        ocr_json = json.load(f)

    results = {
        "acc": {
            "add": 0,
            "correct": 0,
            "edit": 0,
            "move_old": 0,
            "move_new": 0,
            "remove": 0
        },
        "acc_count": {
            "add": 0,
            "correct": 0,
            "edit": 0,
            "move_old": 0,
            "move_new": 0,
            "remove": 0
        },    
        "ned":{
            "add": 0,
            "correct": 0,
            "edit": 0,
            "move_old": 0,
            "move_new": 0,
            "remove": 0  
        },
        "ned_count":{
            "add": 0,
            "correct": 0,
            "edit": 0,
            "move_old": 0,
            "move_new": 0,
            "remove": 0  
        }
    }
    key_list = ['add', "correct", "move_old", "move_new", "remove", "edit"]

    label_list = ocr_json['label_list']
    per_image_result = ocr_json['per_image_result']

    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]
        if label not in key_list:
            print(label)
            raise
        try:
            if metric['acc'] <= 1 and metric['acc'] >= 0: 
                results['acc'][label] += metric['acc']
                results['acc_count'][label] += 1

            if metric['ned'] <= 1 and metric['ned'] >= 0: 
                results['ned'][label] += metric['ned']
                results['ned_count'][label] += 1
        except Exception as e:
            print(e)
            continue
    
    for key in key_list:
        if results['acc_count'][key] != 0:
            results['acc'][key] /= results['acc_count'][key]
        else:
            results['acc'][key] = 0
        
        if results['ned_count'][key] != 0:
            results['ned'][key] /= results['ned_count'][key]
        else:
            results['ned'][key] = 0

    return results


def calc_hpsv3(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    results = {
        "edit_subtype": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,

            "correct": 0,
            "move": 0,
            "rotate": 0,
            "remove": 0
        },
        "count": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,

            "correct": 0,
            "move": 0,
            "rotate": 0,
            "remove": 0
        },
    }
    key_list = list(results['edit_subtype'])

    label_list = datas['label_list']
    per_image_result = datas['per_sample_scores']

    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]
        if label == "size":
            continue

        if label not in key_list:
            print(label)
            raise

        results['edit_subtype'][label] += metric
        results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results


def calc_siglip2_taii(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    results = {
        "edit_subtype": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "correct": 0,
            "move_old": 0,
            "move_new": 0,
            "rotate": 0,
            "remove": 0
        },
        "count": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "correct": 0,
            "move_old": 0,
            "move_new": 0,
            "rotate": 0,
            "remove": 0
        },
    }
    key_list = list(results['edit_subtype'])

    label_list = datas['label_list']
    per_image_result = datas['per_sample_scores']

    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]

        if label == "size":
            continue

        if label not in key_list:
            print(label)
            raise

        results['edit_subtype'][label] += metric
        results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results

def calc_siglip2_tait(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    results = {
        "edit_subtype": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit_old": 0,
            "edit_new": 0,
            "weight": 0,
            "correct": 0,
            "move_old": 0,
            "move_new": 0,
            "rotate": 0,
            "remove": 0
        },
        "count": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit_old": 0,
            "edit_new": 0,
            "weight": 0,
            "correct": 0,
            "move_old": 0,
            "move_new": 0,
            "rotate": 0,
            "remove": 0
        },
    }
    key_list = list(results['edit_subtype'])

    label_list = datas['label_list']
    per_image_result = datas['per_sample_scores']

    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]

        if label == "size":
            continue

        if label not in key_list:
            print(label)
            raise

        results['edit_subtype'][label] += metric
        results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results



def calc_siglip2_wit(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    results = {
        "edit_subtype": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "correct": 0,
            "move": 0,
            "rotate": 0,
            "remove": 0
        },
        "count": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "correct": 0,
            "move": 0,
            "rotate": 0,
            "remove": 0
        },
    }
    key_list = list(results['edit_subtype'])

    label_list = datas['label_list']
    per_image_result = datas['per_sample_scores']

    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]

        if label == "size":
            continue

        if label not in key_list:
            print(label)
            raise

        results['edit_subtype'][label] += metric
        results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results



def calc_traditional_metrics(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    results = {
        "LPIPS":{
            "edit_subtype": {
                "add": 0,
                "color": 0,
                "gradient": 0,
                "texture": 0,
                "font": 0,
                "edit": 0,
                "weight": 0,
                "size": 0,
                "correct": 0,
                "move": 0,
                "rotate": 0,
                "remove": 0
            },
            "count": {
                "add": 0,
                "color": 0,
                "gradient": 0,
                "texture": 0,
                "font": 0,
                "edit": 0,
                "weight": 0,
                "size": 0,
                "correct": 0,
                "move": 0,
                "rotate": 0,
                "remove": 0
            }
        },
        "SSIM":{
            "edit_subtype": {
                "add": 0,
                "color": 0,
                "gradient": 0,
                "texture": 0,
                "font": 0,
                "edit": 0,
                "weight": 0,
                "size": 0,
                "correct": 0,
                "move": 0,
                "rotate": 0,
                "remove": 0
            },
            "count": {
                "add": 0,
                "color": 0,
                "gradient": 0,
                "texture": 0,
                "font": 0,
                "edit": 0,
                "weight": 0,
                "size": 0,
                "correct": 0,
                "move": 0,
                "rotate": 0,
                "remove": 0
            }
        },
        "PSNR":{
            "edit_subtype": {
                "add": 0,
                "color": 0,
                "gradient": 0,
                "texture": 0,
                "font": 0,
                "edit": 0,
                "weight": 0,
                "size": 0,
                "correct": 0,
                "move": 0,
                "rotate": 0,
                "remove": 0
            },
            "count": {
                "add": 0,
                "color": 0,
                "gradient": 0,
                "texture": 0,
                "font": 0,
                "edit": 0,
                "weight": 0,
                "size": 0,
                "correct": 0,
                "move": 0,
                "rotate": 0,
                "remove": 0
            }
        },
        "FID": 0
    }
    label_list = datas['label_list']

    # PSNR
    key_list = list(results['PSNR']['edit_subtype'])
    for i in range(len(label_list)):
        metric = datas['PSNR_list'][i]
        label = label_list[i].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        results['PSNR']['edit_subtype'][label] += metric
        results['PSNR']['count'][label] += 1

    # LPIPS
    key_list = list(results['LPIPS']['edit_subtype'])
    for i in range(len(label_list)):
        metric = datas['LPIPS_list'][i]
        label = label_list[i].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        results['LPIPS']['edit_subtype'][label] += metric
        results['LPIPS']['count'][label] += 1

    # SSIM
    key_list = list(results['SSIM']['edit_subtype'])
    for i in range(len(label_list)):
        metric = datas['SSIM_list'][i]
        label = label_list[i].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        results['SSIM']['edit_subtype'][label] += metric
        results['SSIM']['count'][label] += 1


    results['FID'] = datas['FID']

    
    for key in ['PSNR', 'SSIM', 'LPIPS']:
        key_list = list(results[key]['edit_subtype'].keys())
        for k in key_list:
            if results[key]['count'][k] != 0:
                results[key]['edit_subtype'][k] /= results[key]['count'][k]
            else:
                results[key]['edit_subtype'][k] = 0
        

    return results


def calc_gpt_tasc(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    results = {
        "edit_subtype": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "size": 0,
            "correct": 0,
            "move_remove": 0,
            "move_add": 0,
            "rotate": 0,
            "remove": 0
        },
        "count": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "size": 0,
            "correct": 0,
            "move_remove": 0,
            "move_add": 0,
            "rotate": 0,
            "remove": 0
        },
    }
    key_list = list(results['edit_subtype'])

    for data in datas:
        metric = int(data['gpt_judgement']["score"])
        label = data['label'].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        results['edit_subtype'][label] += metric
        results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results


def calc_gpt_wsc(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    results = {
        "edit_subtype": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "size": 0,
            "correct": 0,
            "move": 0,
            "rotate": 0,
            "remove": 0
        },
        "count": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "size": 0,
            "correct": 0,
            "move": 0,
            "rotate": 0,
            "remove": 0
        },
    }
    key_list = list(results['edit_subtype'])

    for data in datas:
        metric = int(data['gpt_judgement']["score"])
        label = data['label'].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        results['edit_subtype'][label] += metric
        results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results

def calc_gpt_wpq(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    results = {
        "edit_subtype": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "size": 0,
            "correct": 0,
            "move": 0,
            "rotate": 0,
            "remove": 0
        },
        "count": {
            "add": 0,
            "color": 0,
            "gradient": 0,
            "texture": 0,
            "font": 0,
            "edit": 0,
            "weight": 0,
            "size": 0,
            "correct": 0,
            "move": 0,
            "rotate": 0,
            "remove": 0
        },
    }
    key_list = list(results['edit_subtype'])

    for data in datas:
        metric = int(data['gpt_judgement']["score"])
        label = data['label'].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        results['edit_subtype'][label] += metric
        results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.path.join(REPO_ROOT, "results"))
    args = parser.parse_args()

    result_dir = args.save_dir
    total_json = {}

    print('start get ocr result')
    data_path = os.path.join(result_dir, 'deepseek-ocr_acc_ned.json')
    total_json['ocr'] = calc_ocr(data_path)

    print('start get hpsv3 result')
    data_path = os.path.join(result_dir, 'hpsv3_whole.json')
    total_json['hpsv3_whole'] = calc_hpsv3(data_path)

    print('start get siglip2 taii result')
    data_path = os.path.join(result_dir, 'siglip2_target_area_img_img.json')
    total_json['siglip2_taii'] = calc_siglip2_taii(data_path)

    print('start get siglip2 tait result')
    data_path = os.path.join(result_dir, 'siglip2_target_area_img_txt.json')
    total_json['siglip2_tait'] = calc_siglip2_tait(data_path)

    print('start get siglip2 wit result')
    data_path = os.path.join(result_dir, 'siglip2_whole_image_img_txt.json')
    total_json['siglip2_wit'] = calc_siglip2_wit(data_path)

    print('start get fid, ssim, psnr, lpips background results')
    data_path = os.path.join(result_dir, 'traditional_metrics.json')
    total_json['traditional_metrics'] = calc_traditional_metrics(data_path)

    print('start gpt tasc results')
    data_path = os.path.join(result_dir, 'gpt_target_area_semantic_accuracy.json')
    total_json['gpt_tasc'] = calc_gpt_tasc(data_path)

    print('start gpt wsc results')
    data_path = os.path.join(result_dir, 'gpt_whole_semantic_accuracy.json')
    total_json['gpt_wsc'] = calc_gpt_wsc(data_path)

    print('start gpt wpq results')
    data_path = os.path.join(result_dir, 'gpt_whole_perceptual_quality.json')
    total_json['gpt_wpq'] = calc_gpt_wpq(data_path)


    final_result_save_path = os.path.join(result_dir, 'all_results.json')
    with open(final_result_save_path, 'w', encoding='utf-8') as f:
        json.dump(total_json, f, ensure_ascii=False, indent=4)
