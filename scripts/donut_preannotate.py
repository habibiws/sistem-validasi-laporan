# scripts/donut_preannotate.py
import os
import json
import torch
import argparse
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import sys

# Menambahkan path root proyek agar bisa impor dari folder lain jika dibutuhkan
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def donut_preannotate(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = sequence.strip()

    try:
        structured_data = processor.token2json(sequence)
        print(f"    - Berhasil mengekstrak {len(structured_data.keys())} item dari {os.path.basename(image_path)}")
        return structured_data
    except Exception as e:
        print(f"    - [!] Gagal mem-parsing output untuk {os.path.basename(image_path)}: {e}")
        return {}

def main():
    # Menentukan path secara otomatis berdasarkan lokasi skrip
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_IMAGE_DIR = os.path.join(project_root, "data_preparation", "01_raw_images")
    DEFAULT_OUTPUT_DIR = os.path.join(project_root, "data_preparation", "02_pre_annotations")
    
    parser = argparse.ArgumentParser(description="Pra-anotasi cerdas menggunakan Donut.")
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help=f"Direktori gambar. Default: {DEFAULT_IMAGE_DIR}")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help=f"Direktori output. Default: {DEFAULT_OUTPUT_DIR}")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Memuat model Donut (mungkin butuh beberapa saat saat pertama kali)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Menggunakan device: {device}")
    
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2").to(device)

    all_results = []
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nDitemukan {len(image_files)} gambar. Memulai proses pra-anotasi...")

    for filename in image_files:
        image_path = os.path.join(args.image_dir, filename)
        try:
            donut_output = donut_preannotate(image_path, processor, model, device)
            all_results.append({"file_name": filename, "donut_extraction": donut_output})
        except Exception as e:
            print(f"  - [!!!] Error fatal saat memproses {filename}: {e}")
            
    output_path = os.path.join(args.output_dir, "donut_preannotations.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("\nPra-anotasi Selesai! ✅")
    print(f"Hasil ekstraksi teks disimpan di: {output_path}")

if __name__ == "__main__":
    main()