# create_annotation_data.py
import json
import argparse
import fitz
from PIL import Image
import re

from backend.konteks_extractor import load_model, analisis_halaman_dengan_layoutlmv3

# Ganti fungsi ini di create_annotation_data.py

def merge_tokens_to_words(tokens: list, y_tolerance: int = 5, x_tolerance_ratio: float = 0.5) -> list:
    """
    Menggabungkan token yang berdekatan menjadi kata utuh dengan metode sorting yang disempurnakan.
    """
    if not tokens:
        return []

    # --- PERBAIKAN UTAMA ADA DI SINI ---
    # Kita tidak mengelompokkan per baris secara manual, tapi menyempurnakan kriteria sorting.
    # `round(t['box'][1] / 10)` akan menganggap token dalam rentang 10 piksel vertikal sebagai satu baris.
    tokens.sort(key=lambda t: (round(t['box'][1] / 10), t['box'][0]))

    words = []
    current_word_tokens = []

    if not tokens:
        return []

    for token in sorted(tokens, key=lambda t: (round(t['box'][1] / 10), t['box'][0])):
        token_text = token['token'].replace(' ', ' ')
        
        if not current_word_tokens:
            current_word_tokens.append(token)
            continue

        last_token = current_word_tokens[-1]
        
        # Cek kedekatan di baris yang sama
        is_same_line = abs(last_token['box'][1] - token['box'][1]) < y_tolerance

        # Cek kedekatan horizontal untuk digabung menjadi satu kata
        last_token_height = last_token['box'][3] - last_token['box'][1]
        x_tolerance = last_token_height * x_tolerance_ratio if last_token_height > 0 else 5
        is_adjacent = (token['box'][0] - last_token['box'][2]) < x_tolerance
        
        if is_same_line and is_adjacent and not token_text.startswith(' '):
            current_word_tokens.append(token)
        else:
            # Kata sebelumnya selesai, simpan
            word_text = "".join(t['token'] for t in current_word_tokens).replace(' ', ' ').strip()
            if word_text:
                min_x0 = min(t['box'][0] for t in current_word_tokens)
                min_y0 = min(t['box'][1] for t in current_word_tokens)
                max_x1 = max(t['box'][2] for t in current_word_tokens)
                max_y1 = max(t['box'][3] for t in current_word_tokens)
                
                guess = "OTHER"
                if ":" in word_text: guess = "KEY"
                elif re.fullmatch(r'[\d,.-]+', word_text) or re.search(r'\d{1,2}\s+\w+\s+\d{4}', word_text, re.IGNORECASE): guess = "VALUE"
                
                words.append({"word": word_text, "box": [min_x0, min_y0, max_x1, max_y1], "label": guess})
            
            current_word_tokens = [token]

    # Proses kata terakhir
    if current_word_tokens:
        word_text = "".join(t['token'] for t in current_word_tokens).replace(' ', ' ').strip()
        if word_text:
            min_x0 = min(t['box'][0] for t in current_word_tokens)
            min_y0 = min(t['box'][1] for t in current_word_tokens)
            max_x1 = max(t['box'][2] for t in current_word_tokens)
            max_y1 = max(t['box'][3] for t in current_word_tokens)
            guess = "OTHER"
            if ":" in word_text: guess = "KEY"
            elif re.fullmatch(r'[\d,.-]+', word_text) or re.search(r'\d{1,2}\s+\w+\s+\d{4}', word_text, re.IGNORECASE): guess = "VALUE"
            words.append({"word": word_text, "box": [min_x0, min_y0, max_x1, max_y1], "label": guess})

    return words

def main():
    parser = argparse.ArgumentParser(description="Mengekstrak dan menyiapkan data anotasi dari satu halaman PDF.")
    parser.add_argument("--pdf", required=True, help="Path ke file PDF sumber.")
    parser.add_argument("--halaman", required=True, type=int, help="Nomor halaman yang akan diproses (dimulai dari 1).")
    parser.add_argument("--output", required=True, help="Path untuk menyimpan file JSON yang siap dianotasi (level-kata).")
    args = parser.parse_args()

    # 1. Muat Model
    print("Memuat model AI...")
    load_model()

    # 2. Render PDF ke Gambar
    print(f"Membuka PDF '{args.pdf}' dan merender halaman {args.halaman}...")
    try:
        doc = fitz.open(args.pdf)
        page = doc.load_page(args.halaman - 1)
        pix = page.get_pixmap(dpi=300) # Tingkatkan DPI untuk kualitas lebih baik
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
    except Exception as e:
        print(f"[ERROR] Gagal membuka atau merender PDF: {e}")
        return

    # 3. Ekstrak Token
    print("Menganalisis gambar untuk mendapatkan token mentah...")
    hasil_analisis = analisis_halaman_dengan_layoutlmv3(image)
    raw_tokens = hasil_analisis.get("hasil_analisis_kontekstual", [])

    # 4. Gabungkan Token menjadi Kata
    print("Menggabungkan token menjadi kata dan menebak label awal...")
    words_for_annotation = merge_tokens_to_words(raw_tokens)

    # 5. Simpan Hasil
    print(f"Menyimpan {len(words_for_annotation)} kata ke file output: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(words_for_annotation, f, indent=4, ensure_ascii=False)
        
    print("\nSelesai! File yang siap dianotasi telah dibuat.")

if __name__ == "__main__":
    main()