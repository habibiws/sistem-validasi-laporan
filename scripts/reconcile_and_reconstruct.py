# scripts/reconcile_and_reconstruct.py
import json
import os
import argparse
from PIL import Image
import cv2
import numpy as np
from transformers import LayoutLMv3Processor

def remove_table_lines(pil_image: Image.Image) -> Image.Image:
    open_cv_image = np.array(pil_image)
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255, 255, 255), 3)
    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255, 255, 255), 3)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Rekonsiliasi anotasi Label Studio dengan OCR asli LayoutLM.")
    parser.add_argument("--ls_export_path", required=True, help="Path ke file JSON yang diekspor dari Label Studio.")
    parser.add_argument("--image_dir", default=os.path.join(project_root, "data_preparation", "01_raw_images"), help="Direktori berisi file gambar asli.")
    parser.add_argument("--output_dir", default=os.path.join(project_root, "data_preparation", "03_training_data"), help="Direktori untuk menyimpan data training final.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Memuat LayoutLMv3Processor untuk OCR asli...")
    processor = LayoutLMv3Processor.from_pretrained("models/layoutlmv3-base", apply_ocr=True)

    print(f"Membaca 'peta kebenaran' dari: {args.ls_export_path}")
    with open(args.ls_export_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    for task in tasks:
        path_from_json = task['data']['image']
        base_filename = os.path.basename(path_from_json)
        try:
            image_filename = base_filename.split('-', 1)[1]
        except IndexError:
            image_filename = base_filename

        image_path = os.path.join(args.image_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"[Peringatan] Gambar {image_filename} (dari path {base_filename}) tidak ditemukan, melewati.")
            continue
            
        print(f"-> Memproses {image_filename}...")
        image = Image.open(image_path).convert("RGB")
        
        cleaned_image = remove_table_lines(image)
        encoding = processor(cleaned_image, return_offsets_mapping=True)
        
        # --- PERBAIKAN DI SINI ---
        # Langsung gunakan list, tanpa .tolist()
        native_boxes = encoding['bbox'][0]
        native_token_ids = encoding['input_ids'][0]
        
        native_tokens = processor.tokenizer.convert_ids_to_tokens(native_token_ids)
        
        human_annotations = []
        for ann in task['annotations'][0]['result']:
            val = ann['value']
            human_annotations.append({
                'box': [
                    val['x'] / 100.0 * 1000,
                    val['y'] / 100.0 * 1000,
                    (val['x'] + val['width']) / 100.0 * 1000,
                    (val['y'] + val['height']) / 100.0 * 1000
                ],
                'label': val['rectanglelabels'][0]
            })
            
        final_data = []
        for token, box in zip(native_tokens, native_boxes):
            if not any(box): continue
            token_center_x = (box[0] + box[2]) / 2
            token_center_y = (box[1] + box[3]) / 2
            assigned_label = 'OTHER'
            for human_ann in human_annotations:
                h_box = human_ann['box']
                if h_box[0] <= token_center_x <= h_box[2] and h_box[1] <= token_center_y <= h_box[3]:
                    assigned_label = human_ann['label']
                    break
            final_data.append({"word": token, "box": box, "label": assigned_label})
            
        output_filename = os.path.splitext(image_filename)[0] + '.json'
        output_path = os.path.join(args.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

    print("\nRekonsiliasi Selesai! ✅")
    print(f"Dataset training final Anda siap di: {args.output_dir}")

if __name__ == "__main__":
    main()