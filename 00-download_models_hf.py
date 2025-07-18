import json
import os
import shutil
from huggingface_hub import snapshot_download

try:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: 
    CURRENT_DIR = os.getcwd() 
print(f"Script/working directory: {CURRENT_DIR}")

TARGET_MODEL_BASE_DIR = os.path.join(CURRENT_DIR, "models")
print(f"Target model base directory: {TARGET_MODEL_BASE_DIR}")

os.makedirs(TARGET_MODEL_BASE_DIR, exist_ok=True)

def load_modify_save_json(template_filename, output_filename, modifications):
    data = {}
    if os.path.exists(template_filename):
        try:
            with open(template_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Configuration template loaded: {template_filename}")
        except json.JSONDecodeError:
            print(f"JSON template read error: {template_filename}. Starting with empty configuration.")
        except Exception as e:
            print(f"Template load error {template_filename}: {e}. Starting with empty configuration.")
    else:
        print(f"Template file not found: {template_filename}. Starting with empty configuration.")

    for key, value in modifications.items():
        data[key] = value
    print(f"Modification applied: {modifications}")

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Configuration file saved: {output_filename}")
    except Exception as e:
        print(f"Configuration file save error {output_filename}: {e}")

if __name__ == '__main__':

    mineru_patterns = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_hf_small_2503/*",
        "models/OCR/paddleocr_torch/*",
        #"models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    print("Downloading PDF-Extract-Kit models...")
    pdf_kit_download_dir = snapshot_download(
        'opendatalab/PDF-Extract-Kit-1.0',
        allow_patterns=mineru_patterns,
        local_dir=TARGET_MODEL_BASE_DIR,
        local_dir_use_symlinks=False
    )
    final_pdf_kit_model_dir = os.path.join(pdf_kit_download_dir, 'models')
    print(f"PDF-Extract-Kit download location: {pdf_kit_download_dir}")
    print(f"PDF-Extract-Kit valid model path for configuration: {final_pdf_kit_model_dir}")


    layoutreader_target_subdir = os.path.join(TARGET_MODEL_BASE_DIR, "layoutreader")
    os.makedirs(layoutreader_target_subdir, exist_ok=True)

    layoutreader_pattern = [
        "*.json",
        "*.safetensors",
    ]
    print("Downloading LayoutReader models...")
    layoutreader_model_dir = snapshot_download(
        'hantian/layoutreader',
        allow_patterns=layoutreader_pattern,
        local_dir=layoutreader_target_subdir,
        local_dir_use_symlinks=False
    )
    print(f"LayoutReader download location: {layoutreader_model_dir}")


    json_template_file = os.path.join(CURRENT_DIR, 'magic-pdf.template.json')
    config_file_name = 'magic-pdf.json'
    config_file_output_path = os.path.join(CURRENT_DIR, config_file_name)

    json_mods = {
        'models-dir': final_pdf_kit_model_dir,
        'layoutreader-model-dir': layoutreader_model_dir
    }

    print(f"Updating configuration file: {config_file_output_path}")
    load_modify_save_json(json_template_file, config_file_output_path, json_mods)

    print(f"--- Configuration Summary ---")
    print(f"PDF-Extract-Kit model path: {json_mods['models-dir']}")
    print(f"LayoutReader model path: {json_mods['layoutreader-model-dir']}")
    print(f"Configuration file saved: {config_file_output_path}")
    print(f"-----------------------------")