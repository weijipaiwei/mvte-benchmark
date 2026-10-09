import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
from PIL import Image

def filter_data(data_path, save_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for _k in ("original_imgs_root", "edited_imgs_root"):
            if isinstance(data.get(_k), str) and not os.path.isabs(data[_k]):
                data[_k] = os.path.join(REPO_ROOT, data[_k])

    data['edited_imgs_root'] = os.path.join(os.path.dirname(save_path), "edited")
    # drop entries whose edited image was never generated
    filtered_data_list = {}
    for key in data['data_list'].keys():
        filtered_data_list[key] = []
        for sample in data['data_list'][key]:
            edited_path = os.path.join(data['edited_imgs_root'], key, sample[0])
            if os.path.exists(edited_path):
                try:
                    with Image.open(edited_path) as img_src:
                        img_src.convert('RGB')
                    filtered_data_list[key].append(sample)
                except Exception as e:
                    print(e)
                    continue
            
    data['data_list'] = filtered_data_list
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path1', type=str, default=os.path.join(REPO_ROOT, "benchmark", "validation.json"))
    parser.add_argument('--json_path2', type=str, default=None)
    args = parser.parse_args()

    filter_data(args.json_path1, args.json_path2)




