import json
import argparse
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Side, Font
from openpyxl.utils import get_column_letter

def calc_avg(json_data):
    avg_dict = {}

    total_acc = 0
    total_count = 0
    for key in json_data['ocr']['acc'].keys():
        count = json_data['ocr']['acc_count'][key]
        if key not in ['remove', 'move_old']:
            total_acc += json_data['ocr']['acc'][key] * count
        else:
            total_acc += (1-json_data['ocr']['acc'][key]) * count
        total_count += count
    avg_dict['ocr-ta-acc'] =  total_acc / total_count

    total_ned = 0
    total_count = 0
    for key in json_data['ocr']['ned'].keys():
        count = json_data['ocr']['ned_count'][key]
        if key not in ['remove', 'move_old']:
            total_ned += json_data['ocr']['ned'][key] * count
        else:
            total_ned += (1-json_data['ocr']['ned'][key]) * count
        total_count += count
    avg_dict['ocr-ta-ned'] =  total_ned / total_count

    total_hpsv3 = 0
    total_count = 0
    for key in json_data['hpsv3_whole']['edit_subtype'].keys():
        count = json_data['hpsv3_whole']['count'][key]
        total_hpsv3 +=  json_data['hpsv3_whole']['edit_subtype'][key] * count
        total_count += count
    avg_dict['hpsv3-w'] =  total_hpsv3 / total_count

    total_siglip2_taii = 0
    total_count = 0
    for key in json_data['siglip2_taii']['edit_subtype'].keys():
        count = json_data['siglip2_taii']['count'][key]
        total_siglip2_taii +=  json_data['siglip2_taii']['edit_subtype'][key] * count
        total_count += count
    avg_dict['siglip2-taii'] =  total_siglip2_taii / total_count


    total_siglip2_tait = 0
    total_count = 0
    for key in json_data['siglip2_tait']['edit_subtype'].keys():
        count = json_data['siglip2_tait']['count'][key]
        total_siglip2_tait +=  json_data['siglip2_tait']['edit_subtype'][key] * count
        total_count += count
    avg_dict['siglip2-tait'] =  total_siglip2_tait / total_count


    total_siglip2_wit = 0
    total_count = 0
    for key in json_data['siglip2_wit']['edit_subtype'].keys():
        count = json_data['siglip2_wit']['count'][key]
        total_siglip2_wit +=  json_data['siglip2_wit']['edit_subtype'][key] * count
        total_count += count
    avg_dict['siglip2-wit'] =  total_siglip2_wit / total_count

    avg_dict['fid-bk'] = json_data['traditional_metrics']['FID']

    total_ssim = 0
    total_count = 0
    for key in json_data['traditional_metrics']['SSIM']['edit_subtype'].keys():
        count = json_data['traditional_metrics']['SSIM']['count'][key]
        total_ssim += json_data['traditional_metrics']['SSIM']['edit_subtype'][key] * count
        total_count += count
    avg_dict['ssim-bk'] =  total_ssim / total_count


    total_psnr = 0
    total_count = 0
    for key in json_data['traditional_metrics']['PSNR']['edit_subtype'].keys():
        count = json_data['traditional_metrics']['PSNR']['count'][key]
        total_psnr += json_data['traditional_metrics']['PSNR']['edit_subtype'][key] * count
        total_count += count
    avg_dict['psnr-bk'] =  total_psnr / total_count


    total_lpips = 0
    total_count = 0
    for key in json_data['traditional_metrics']['LPIPS']['edit_subtype'].keys():
        count = json_data['traditional_metrics']['LPIPS']['count'][key]
        total_lpips += json_data['traditional_metrics']['LPIPS']['edit_subtype'][key] * count
        total_count += count
    avg_dict['lpips-bk'] =  total_lpips / total_count


    total_gpt_tasc = 0
    total_count = 0
    for key in json_data['gpt_tasc']['edit_subtype'].keys():
        count = json_data['gpt_tasc']['count'][key]
        total_gpt_tasc += json_data['gpt_tasc']['edit_subtype'][key] * count
        total_count += count
    avg_dict['gpt-tasc'] =  total_gpt_tasc / total_count


    total_gpt_wsc = 0
    total_count = 0
    for key in json_data['gpt_wsc']['edit_subtype'].keys():
        count = json_data['gpt_wsc']['count'][key]
        total_gpt_wsc += json_data['gpt_wsc']['edit_subtype'][key] * count
        total_count += count
    avg_dict['gpt-wsc'] =  total_gpt_wsc / total_count


    total_gpt_wpq = 0
    total_count = 0
    for key in json_data['gpt_wpq']['edit_subtype'].keys():
        count = json_data['gpt_wpq']['count'][key]
        total_gpt_wpq += json_data['gpt_wpq']['edit_subtype'][key] * count
        total_count += count
    avg_dict['gpt-bkpq'] =  total_gpt_wpq / total_count

    return  avg_dict

def fill_sheet_data(ws, json_data, avg_data):
    """
    Revised helper: fills a worksheet; does not create the Workbook or save the file.
    """
    # ==========================================
    # 1. set the header
    # ==========================================
    
    # first header row
    headers_row1 = {
        "A": "metrics\\edit type",
        "B": "ADD",
        "C": "COLOR", # C-F
        "G": "EDIT",  # G-J
        "K": "MOVE",  # K-L
        "M": "REMOVE", # M
        "N": "AVG"    # N (new AVG column)
    }
    
    # second header row
    headers_row2 = [
        "",             # A
        "add 100",      # B
        "color 33",     # C
        "gradient 37",  # D
        "texture 30",   # E
        "font 10",      # F
        "edit 100",     # G
        "weight 10",    # H
        "size 10",      # I
        "correct 5",    # J
        "move 100",     # K
        "rotate 10",    # L
        "remove 100",   # M
        "AVG"           # N
    ]

    # write the first row
    for col, value in headers_row1.items():
        ws[f"{col}1"] = value
    
    # write the second row
    for i, value in enumerate(headers_row2):
        if i > 0: # skip column A
            ws.cell(row=2, column=i+1, value=value)

    # merge cells (row 1)
    ws.merge_cells('A1:A2') # metrics\edit type
    ws.merge_cells('C1:F1') # COLOR
    ws.merge_cells('G1:J1') # EDIT
    ws.merge_cells('K1:L1') # MOVE
    ws.merge_cells('N1:N2') # AVG (vertical merge)
    
    # ==========================================
    # 2. helpers and configuration
    # ==========================================
    
    col_map = {
        2: 'add', 3: 'color', 4: 'gradient', 5: 'texture', 6: 'font',
        7: 'edit', 8: 'weight', 9: 'size', 10: 'correct', 
        11: 'move', 12: 'rotate', 13: 'remove'
    }

    def fmt(val):
        if val is None or val == "": 
            return "-"
        if isinstance(val, (int, float)):
            return round(val, 4)
        return val

    # handle the "old / new" format
    def get_split_val(data_dict, key_prefix):
        old = data_dict.get(f"{key_prefix}_old")
        new = data_dict.get(f"{key_prefix}_new")
        
        if old is None and f"{key_prefix}_remove" in data_dict:
             old = data_dict.get(f"{key_prefix}_remove")
             new = data_dict.get(f"{key_prefix}_add")
        
        if old is not None and new is not None:
            return f"{fmt(old)} / {fmt(new)}"
        return None

    # row configuration
    rows_config = [
        {"name": "ocr-ta-acc",   "source": json_data['ocr']['acc'], "type": "split_move"},
        {"name": "ocr-ta-ned",   "source": json_data['ocr']['ned'], "type": "split_move"},
        {"name": "hpsv3-w",      "source": json_data['hpsv3_whole']['edit_subtype'], "type": "normal"},
        {"name": "siglip2-taii", "source": json_data['siglip2_taii']['edit_subtype'], "type": "split_move"}, 
        {"name": "siglip2-tait", "source": json_data['siglip2_tait']['edit_subtype'], "type": "split_all"},
        {"name": "siglip2-wit",  "source": json_data['siglip2_wit']['edit_subtype'], "type": "normal"},
        {"name": "fid-bk",       "source": json_data['traditional_metrics']['FID'], "type": "fid_special"},
        {"name": "ssim-bk",      "source": json_data['traditional_metrics']['SSIM']['edit_subtype'], "type": "normal"},
        {"name": "psnr-bk",      "source": json_data['traditional_metrics']['PSNR']['edit_subtype'], "type": "normal"},
        {"name": "lpips-bk",     "source": json_data['traditional_metrics']['LPIPS']['edit_subtype'], "type": "normal"},
        {"name": "gpt-tasc",     "source": json_data['gpt_tasc']['edit_subtype'], "type": "gpt_tasc"},
        {"name": "gpt-wsc",      "source": json_data['gpt_wsc']['edit_subtype'], "type": "normal"},
        {"name": "gpt-bkpq",     "source": json_data['gpt_wpq']['edit_subtype'], "type": "normal"},
    ]

    # ==========================================
    # 3. fill in the data
    # ==========================================
    
    current_row = 3
    
    for row_cfg in rows_config:
        row_name = row_cfg['name']
        data = row_cfg['source']
        row_type = row_cfg['type']
        
        ws.cell(row=current_row, column=1, value=row_name)
        
        # --- handle the AVG data ---
        # row mean; None when absent (e.g. the special FID row)
        avg_val = avg_data.get(row_name)

        # FID special row
        if row_type == "fid_special":
            # merge columns B-M
            ws.merge_cells(start_row=current_row, start_column=2, end_row=current_row, end_column=13)
            fid_val = fmt(data) if data is not None else "-"
            cell = ws.cell(row=current_row, column=2, value=fid_val)
            cell.alignment = Alignment(horizontal='center', vertical='center')
            
            # the AVG column of the FID row may simply repeat the global value
            ws.cell(row=current_row, column=14, value=fid_val) 
            
            current_row += 1
            continue

        # iterate over columns B(2) to M(13)
        for col_idx in range(2, 14):
            key = col_map[col_idx]
            cell_val = "-" 
            raw_val = data.get(key)
            
            if row_type == "normal":
                if raw_val is not None: cell_val = fmt(raw_val)
            elif row_type == "split_move":
                if key == "move":
                    split_val = get_split_val(data, "move")
                    if split_val: cell_val = split_val
                    elif raw_val is not None: cell_val = fmt(raw_val)
                else:
                    if raw_val is not None: cell_val = fmt(raw_val)
            elif row_type == "split_all":
                if key == "edit":
                    split_val = get_split_val(data, "edit")
                    if split_val: cell_val = split_val
                    elif raw_val is not None: cell_val = fmt(raw_val)
                elif key == "move":
                    split_val = get_split_val(data, "move")
                    if split_val: cell_val = split_val
                    elif raw_val is not None: cell_val = fmt(raw_val)
                else:
                    if raw_val is not None: cell_val = fmt(raw_val)
            elif row_type == "gpt_tasc":
                if key == "move":
                    val_rem = data.get("move_remove")
                    val_add = data.get("move_add")
                    if val_rem is not None and val_add is not None:
                          cell_val = f"{fmt(val_rem)} / {fmt(val_add)}"
                else:
                    if raw_val is not None: cell_val = fmt(raw_val)

            ws.cell(row=current_row, column=col_idx, value=cell_val)
        
        # --- write the AVG column (column 14) ---
        ws.cell(row=current_row, column=14, value=fmt(avg_val))
            
        current_row += 1

    # ==========================================
    # 4. styling
    # ==========================================
    
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), 
                         top=Side(style='thin'), bottom=Side(style='thin'))
    center_align = Alignment(horizontal='center', vertical='center', wrap_text=True)
    header_font = Font(bold=True, name='Times New Roman')
    body_font = Font(name='Times New Roman')

    # extend the range: max_col up to 14
    for row in ws.iter_rows(min_row=1, max_row=current_row-1, min_col=1, max_col=14):
        for cell in row:
            cell.border = thin_border
            cell.alignment = center_align
            if cell.row <= 2:
                cell.font = header_font
            else:
                cell.font = body_font

    ws.column_dimensions['A'].width = 15
    for col_idx in range(2, 15): # 2 to 14
        ws.column_dimensions[get_column_letter(col_idx)].width = 13


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.path.join(REPO_ROOT, "results"))
    args = parser.parse_args()

    # path of the final aggregated Excel file
    final_output_file = os.path.join(args.save_dir, 'all_results_combined_language.xlsx')
    
    # 1. initialise the single Workbook
    wb = Workbook()
    # delete the default "Sheet"
    default_ws = wb.active
    wb.remove(default_ws)


    # ==========================
    # handle the "cn" and "en" data
    # ==========================
    json_rank_path = os.path.join(args.save_dir, 'all_results_language.json')
    if os.path.exists(json_rank_path):
        with open(json_rank_path, 'r', encoding='utf-8') as f:
            json_rank_data = json.load(f)
        
        levels = ['cn', 'en']
        for level in levels:
            if level in json_rank_data:
                # create the corresponding sheet
                ws_level = wb.create_sheet(level)
                
                # fetch the data of this difficulty level
                level_data = json_rank_data[level]
                
                # compute the mean
                avg_level = calc_avg(level_data)
                
                # fill in the data
                fill_sheet_data(ws_level, level_data, avg_level)
    else:
        print(f"Warning: {json_rank_path} not found.")

    # ==========================
    # save the file
    # ==========================
    wb.save(final_output_file)
    print(f"merged Excel file written: {final_output_file}")