# backend/konteks_extractor.py

from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch

# Variabel global untuk menampung model dan prosesor yang sudah dimuat
# Ini memastikan kita hanya memuat model ke memori satu kali.
MODEL = None
PROCESSOR = None

def load_model():
    """
    Memuat model LayoutLMv3 dan prosesor ke dalam variabel global.
    Fungsi ini akan dipanggil sekali saat server FastAPI dimulai.
    """
    global MODEL, PROCESSOR

    if MODEL is None:
        MODEL_NAME = "layoutlmv3-base"
        print(f"Memuat model AI '{MODEL_NAME}' ke memori...")
        
        PROCESSOR = LayoutLMv3Processor.from_pretrained(MODEL_NAME)
        MODEL = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME)
        
        print("Model AI berhasil dimuat dan siap digunakan.")

def parse_predictions(predictions, tokens, boxes, id2label):
    """

    Mengubah output mentah dari model menjadi daftar key-value pair yang lebih bersih.
    """
    # ... (Fungsi ini bisa dibuat lebih kompleks sesuai kebutuhan,
    # saat ini ia akan mengembalikan hasil mentah yang terstruktur)
    
    results = []
    for token, box, pred_id in zip(tokens, tokens, predictions):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        
        results.append({
            "token": token,
            "label": id2label[pred_id],
            "box": box
        })
    return results

# Di dalam file: backend/konteks_extractor.py

def analisis_halaman_dengan_layoutlmv3(image: Image.Image) -> dict:
    """
    Menganalisis satu gambar halaman menggunakan model LayoutLMv3 yang sudah dimuat.
    """
    global MODEL, PROCESSOR

    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("Model belum dimuat. Jalankan load_model() terlebih dahulu.")

    # --- PERBAIKAN DI SINI ---
    # Tambahkan truncation=True untuk menangani halaman dengan teks yang sangat panjang
    encoding = PROCESSOR(image, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = MODEL(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tokens = PROCESSOR.tokenizer.convert_ids_to_tokens(encoding.input_ids.squeeze().tolist())
    boxes = encoding.bbox.squeeze().tolist()

    parsed_results = []
    for token, box, pred_id in zip(tokens, boxes, predictions):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        
        parsed_results.append({
            "token": token,
            "label": MODEL.config.id2label[pred_id],
            "box": box
        })

    return {"hasil_analisis_kontekstual": parsed_results}