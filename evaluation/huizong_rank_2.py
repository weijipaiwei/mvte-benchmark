import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import argparse


def sum_dict(data):
    sum_ = 0
    for k in data.keys():
        sum_ += int(data[k])
    return sum_


def calc_ocr(ocr_json_path, level_json, level):
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
    key_map = {
        'correct': 'edit_image',
        'move_new': 'move_image',
        'remove': 'remove_image',
        'move_old': 'move_image',
        'add': 'add_image',
        'edit': 'edit_image'
    }

    label_list = ocr_json['label_list']
    per_image_result = ocr_json['per_image_result']

    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]

        file_name = label_list[i].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):
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


def calc_hpsv3(json_path, level_json, level):
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

    key_map = {
        "add": 'add_image',
        "color": 'color_image',
        "gradient": 'color_image',
        "texture": 'color_image',
        "font": 'edit_image',
        "edit": 'edit_image',
        "weight": 'edit_image',
        "correct": 'edit_image',
        "move": 'move_image',
        "rotate": 'move_image',
        "remove": 'remove_image'
    }

    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]
        if label == "size":
            continue

        if label not in key_list:
            print(label)
            raise

        file_name = label_list[i].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):

            results['edit_subtype'][label] += metric
            results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results


def calc_siglip2_taii(json_path, level_json, level):
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

    key_map = {
        "add": 'add_image',
        "color": 'color_image',
        "gradient": 'color_image',
        "texture": 'color_image',
        "font": 'edit_image',
        "edit": 'edit_image',
        "weight": 'edit_image',
        "correct": 'edit_image',
        "move_old": 'move_image',
        "move_new": 'move_image',
        "rotate": 'move_image',
        "remove": 'remove_image'
    }


    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]

        if label == "size":
            continue

        if label not in key_list:
            print(label)
            raise

        file_name = label_list[i].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):

            results['edit_subtype'][label] += metric
            results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results

def calc_siglip2_tait(json_path, level_json, level):
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

    key_map = {
        "add": 'add_image',
        "color": 'color_image',
        "gradient": 'color_image',
        "texture": 'color_image',
        "font": 'edit_image',
        "edit_old": 'edit_image',
        "edit_new": 'edit_image',
        "weight": 'edit_image',
        "correct": 'edit_image',
        "move_old": 'move_image',
        "move_new": 'move_image',
        "rotate": 'move_image',
        "remove": 'remove_image'
    }

    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]

        if label == "size":
            continue

        if label not in key_list:
            print(label)
            raise

        file_name = label_list[i].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):

            results['edit_subtype'][label] += metric
            results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results



def calc_siglip2_wit(json_path, level_json, level):
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

    key_map = {
        "add": 'add_image',
        "color": 'color_image',
        "gradient": 'color_image',
        "texture": 'color_image',
        "font": 'edit_image',
        "edit": 'edit_image',
        "weight": 'edit_image',
        "correct": 'edit_image',
        "move": 'move_image',
        "rotate": 'move_image',
        "remove": 'remove_image'
    }

    for i in range(len(label_list)):
        metric = per_image_result[i]
        label = label_list[i].split('\\')[0]

        if label == "size":
            continue

        if label not in key_list:
            print(label)
            raise
        file_name = label_list[i].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):
            results['edit_subtype'][label] += metric
            results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results



def calc_traditional_metrics(json_path, level_json, level):
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
    key_map = {
        "add": 'add_image',
        "color": 'color_image',
        "gradient": 'color_image',
        "texture": 'color_image',
        "font": 'edit_image',
        "edit": 'edit_image',
        "weight": 'edit_image',
        "size": 'edit_image',
        "correct": 'edit_image',
        "move": 'move_image',
        "rotate": 'move_image',
        "remove": 'remove_image'
    }
    # PSNR
    key_list = list(results['PSNR']['edit_subtype'])
    for i in range(len(label_list)):
        metric = datas['PSNR_list'][i]
        label = label_list[i].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        file_name = label_list[i].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):

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

        file_name = label_list[i].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):

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

        file_name = label_list[i].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):
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


def calc_gpt_tasc(json_path, level_json, level):
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

    key_map = {
        "add": 'add_image',
        "color": 'color_image',
        "gradient": 'color_image',
        "texture": 'color_image',
        "font": 'edit_image',
        "edit": 'edit_image',
        "weight": 'edit_image',
        "size": 'edit_image',
        "correct": 'edit_image',
        "move_remove": 'move_image',
        "move_add": 'move_image',
        "rotate": 'move_image',
        "remove": 'remove_image'
    }

    for data in datas:
        metric = int(data['gpt_judgement']["score"])
        label = data['label'].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        file_name = data['label'].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):
            results['edit_subtype'][label] += metric
            results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results


def calc_gpt_wsc(json_path, level_json, level):
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

    key_map = {
        "add": 'add_image',
        "color": 'color_image',
        "gradient": 'color_image',
        "texture": 'color_image',
        "font": 'edit_image',
        "edit": 'edit_image',
        "weight": 'edit_image',
        "size": 'edit_image',
        "correct": 'edit_image',
        "move": 'move_image',
        "rotate": 'move_image',
        "remove": 'remove_image'
    }

    for data in datas:
        metric = int(data['gpt_judgement']["score"])
        label = data['label'].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        file_name = data['label'].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):

            results['edit_subtype'][label] += metric
            results['count'][label] += 1

    
    for key in key_list:
        if results['count'][key] != 0:
            results['edit_subtype'][key] /= results['count'][key]
        else:
            results['edit_subtype'][key] = 0
        

    return results

def calc_gpt_wpq(json_path, level_json, level):
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

    key_map = {
        "add": 'add_image',
        "color": 'color_image',
        "gradient": 'color_image',
        "texture": 'color_image',
        "font": 'edit_image',
        "edit": 'edit_image',
        "weight": 'edit_image',
        "size": 'edit_image',
        "correct": 'edit_image',
        "move": 'move_image',
        "rotate": 'move_image',
        "remove": 'remove_image'
    }

    for data in datas:
        metric = int(data['gpt_judgement']["score"])
        label = data['label'].split('\\')[0]

        if label not in key_list:
            print(label)
            raise

        file_name = data['label'].split('\\')[1]
        rank = sum_dict(level_json[key_map[label]][file_name])

        if (level == 'easy' and rank in [4,5]) or (level == 'hard' and rank in [6,7,8,9,10, 11, 12]):

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
    parser.add_argument('--json_path', type=str, default=os.path.join(REPO_ROOT, "benchmark", "validation.json"))
    args = parser.parse_args()

    result_dir = args.save_dir
    with open(args.json_path, 'r', encoding='utf-8') as f:
        valid_json = json.load(f)

    level_dict = {}
    for k in valid_json['data_list'].keys():
        d_list = valid_json['data_list'][k]
        tmp_dict = {}
        for s in d_list:
            tmp_dict[s[0]] = s[6]
        level_dict[k] = tmp_dict

    total_json = {
        'easy': {},
        'hard': {}
    }

    print('start get ocr result')
    data_path = os.path.join(result_dir, 'deepseek-ocr_acc_ned.json')
    total_json['easy']['ocr'] = calc_ocr(data_path, level_dict, 'easy')
    total_json['hard']['ocr'] = calc_ocr(data_path, level_dict, 'hard')



    print('start get hpsv3 result')
    data_path = os.path.join(result_dir, 'hpsv3_whole.json')
    total_json['easy']['hpsv3_whole'] = calc_hpsv3(data_path, level_dict, 'easy')
    total_json['hard']['hpsv3_whole'] = calc_hpsv3(data_path, level_dict, 'hard')

    print('start get siglip2 taii result')
    data_path = os.path.join(result_dir, 'siglip2_target_area_img_img.json')
    total_json['easy']['siglip2_taii'] = calc_siglip2_taii(data_path, level_dict, 'easy')
    total_json['hard']['siglip2_taii'] = calc_siglip2_taii(data_path, level_dict, 'hard')

    print('start get siglip2 tait result')
    data_path = os.path.join(result_dir, 'siglip2_target_area_img_txt.json')
    total_json['easy']['siglip2_tait'] = calc_siglip2_tait(data_path, level_dict, 'easy')
    total_json['hard']['siglip2_tait'] = calc_siglip2_tait(data_path, level_dict, 'hard')

    print('start get siglip2 wit result')
    data_path = os.path.join(result_dir, 'siglip2_whole_image_img_txt.json')
    total_json['easy']['siglip2_wit'] = calc_siglip2_wit(data_path, level_dict, 'easy')
    total_json['hard']['siglip2_wit'] = calc_siglip2_wit(data_path, level_dict, 'hard')

    print('start get fid, ssim, psnr, lpips background results')
    data_path = os.path.join(result_dir, 'traditional_metrics.json')
    total_json['easy']['traditional_metrics'] = calc_traditional_metrics(data_path, level_dict, 'easy')
    total_json['hard']['traditional_metrics'] = calc_traditional_metrics(data_path, level_dict, 'hard')

    print('start gpt tasc results')
    data_path = os.path.join(result_dir, 'gpt_target_area_semantic_accuracy.json')
    total_json['easy']['gpt_tasc'] = calc_gpt_tasc(data_path, level_dict, 'easy')
    total_json['hard']['gpt_tasc'] = calc_gpt_tasc(data_path, level_dict, 'hard')

    print('start gpt wsc results')
    data_path = os.path.join(result_dir, 'gpt_whole_semantic_accuracy.json')
    total_json['easy']['gpt_wsc'] = calc_gpt_wsc(data_path, level_dict, 'easy')
    total_json['hard']['gpt_wsc'] = calc_gpt_wsc(data_path, level_dict, 'hard')

    print('start gpt wpq results')
    data_path = os.path.join(result_dir, 'gpt_whole_perceptual_quality.json')
    total_json['easy']['gpt_wpq'] = calc_gpt_wpq(data_path, level_dict, 'easy')
    total_json['hard']['gpt_wpq'] = calc_gpt_wpq(data_path, level_dict, 'hard')


    final_result_save_path = os.path.join(result_dir, 'all_results_level_2.json')
    with open(final_result_save_path, 'w', encoding='utf-8') as f:
        json.dump(total_json, f, ensure_ascii=False, indent=4)
