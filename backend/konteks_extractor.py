# backend/konteks_extractor.py

from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch

# Variabel global untuk menampung model dan prosesor yang sudah dimuat
MODEL = None
PROCESSOR = None

def load_model():
    """
    Memuat model LayoutLMv3 dan prosesor ke dalam variabel global.
    """
    global MODEL, PROCESSOR

    if MODEL is None:
        # KEMBALI MENGGUNAKAN NAMA MODEL LOKAL ANDA
        MODEL_NAME = "microsoft/layoutlmv3-base" 
        print(f"Memuat model AI '{MODEL_NAME}' dari folder lokal...")
        
        PROCESSOR = LayoutLMv3Processor.from_pretrained(MODEL_NAME)
        MODEL = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME)
        
        print("Model AI berhasil dimuat dan siap digunakan.")


def analisis_halaman_dengan_layoutlmv3(image: Image.Image) -> dict:
    """
    Menganalisis satu gambar halaman menggunakan model LayoutLMv3 yang sudah dimuat.
    """
    global MODEL, PROCESSOR

    if MODEL is None or PROCESSOR is None:
        raise RuntimeError("Model belum dimuat. Jalankan load_model() terlebih dahulu.")

    # Menambahkan truncation=True untuk menangani halaman dengan teks yang sangat panjang
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


def visualisasikan_hasil_analisis(image: Image.Image, hasil_analisis: dict) -> Image.Image:
    """
    Menggambar bounding box dan label, dengan penyesuaian skala koordinat yang benar.
    """
    img_visual = image.copy()
    draw = ImageDraw.Draw(img_visual)
    width, height = img_visual.size

    def unnormalize_box(box, width, height):
        """Mengubah skala koordinat dari 0-1000 ke dimensi piksel gambar."""
        return [
            int(box[0] / 1000.0 * width),
            int(box[1] / 1000.0 * height),
            int(box[2] / 1000.0 * width),
            int(box[3] / 1000.0 * height),
        ]

    label_colors = {
        "QUESTION": "blue",
        "ANSWER": "green",
        "HEADER": "orange",
        "DEFAULT": "red"
    }
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    data_analisis = hasil_analisis.get("hasil_analisis_kontekstual", [])

    for item in data_analisis:
        normalized_box = item["box"]
        label = item["label"]
        
        pixel_box = unnormalize_box(normalized_box, width, height)
        
        color = label_colors.get(label, label_colors["DEFAULT"])
        
        draw.rectangle(pixel_box, outline=color, width=2)
        
        text_position = (pixel_box[0] + 5, pixel_box[1] - 15)
        draw.rectangle([text_position, (text_position[0] + len(label) * 8, text_position[1] + 15)], fill="white")
        draw.text(text_position, label, fill=color, font=font)
        
    return img_visual