# train_model.py (Versi yang Disesuaikan dengan Struktur Folder Baru)
import json
import glob
import random
import torch
import os
from PIL import Image
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score 
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [unique_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [unique_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# --- KONFIGURASI PATH BARU ---
PATH_TO_ANNOTATIONS = "data_preparation/03_training_data/*.json"
BASE_MODEL_PATH = "models/layoutlmv3-base"
NEW_MODEL_PATH = "models/layoutlmv3-finetuned-laporan"
CHECKPOINTS_PATH = "training_output/checkpoints"

# Pastikan folder output ada
os.makedirs(NEW_MODEL_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)


# --- 1. Memuat dan Mempersiapkan Dataset ---
print(f"Mencari semua file anotasi di: {PATH_TO_ANNOTATIONS}")
all_annotation_files = glob.glob(PATH_TO_ANNOTATIONS)
if not all_annotation_files:
    raise ValueError(f"Tidak ada file anotasi .json yang ditemukan! Periksa path: '{PATH_TO_ANNOTATIONS}'")

random.shuffle(all_annotation_files)
print(f"Ditemukan {len(all_annotation_files)} file anotasi.")

# Bagi data menjadi set training dan evaluasi (misal, 80% train, 20% eval)
split_index = int(len(all_annotation_files) * 0.8)
if len(all_annotation_files) < 2: # Jika hanya ada 1 file, gunakan untuk keduanya
    train_files = all_annotation_files
    eval_files = all_annotation_files
else:
    train_files = all_annotation_files[:split_index]
    eval_files = all_annotation_files[split_index:]
print(f"Membagi data: {len(train_files)} untuk training, {len(eval_files)} untuk evaluasi.")

def load_dataset_from_files(files):
    data = []
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    return data

train_data = load_dataset_from_files(train_files)
eval_data = load_dataset_from_files(eval_files)

# Ambil semua label unik dari data training
unique_labels = sorted(list(set(item['label'] for page in train_data for item in page)))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for i, label in enumerate(unique_labels)}
print(f"Label yang akan dilatih: {label2id}")

def create_dataset_dict(data):
    # Menggabungkan token menjadi kata utuh (perbaikan sederhana)
    words_list, boxes_list, labels_list = [], [], []
    for page in data:
        if not page: continue
        words, boxes, labels = [], [], []
        for item in page:
            # Mengganti token khusus tokenizer agar tidak error
            word = item['word'].replace('Ġ', '') 
            if word: # Hanya tambahkan jika kata tidak kosong
                words.append(word)
                boxes.append(item['box'])
                labels.append(label2id[item['label']])
        if words: # Hanya tambahkan halaman jika tidak kosong
            words_list.append(words)
            boxes_list.append(boxes)
            labels_list.append(labels)
            
    return {"words": words_list, "bboxes": boxes_list, "ner_tags": labels_list}

train_dataset = Dataset.from_dict(create_dataset_dict(train_data))
eval_dataset = Dataset.from_dict(create_dataset_dict(eval_data))


# --- 2. Pra-pemrosesan Data untuk Model ---
processor = LayoutLMv3Processor.from_pretrained(BASE_MODEL_PATH, apply_ocr=False)

features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': Sequence(ClassLabel(names=unique_labels)),
})

def preprocess_data(examples):
    # Buat gambar placeholder karena kita fokus pada teks dan layout
    images = [Image.new("RGB", (1000, 1000)) for _ in range(len(examples['words']))] 
    
    encoded_inputs = processor(
        images,
        examples['words'],
        boxes=examples['bboxes'],
        word_labels=examples['ner_tags'],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    return encoded_inputs

train_dataset = train_dataset.map(preprocess_data, batched=True, remove_columns=train_dataset.column_names, features=features)
eval_dataset = eval_dataset.map(preprocess_data, batched=True, remove_columns=eval_dataset.column_names, features=features)


# --- 3. Memuat Model dan Konfigurasi Training ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Akan menggunakan device: {device}")

model = LayoutLMv3ForTokenClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
).to(device)

# Argumen training disesuaikan untuk hemat memori
training_args = TrainingArguments(
    output_dir=CHECKPOINTS_PATH,
    num_train_epochs=100, # Bisa disesuaikan
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./training_logs',
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# --- 4. Mulai Proses Training ---
print("="*50)
print("Memulai proses training model...")
print("="*50)

trainer.train()

# --- 5. Simpan Model Final ---
print("Training selesai. Menyimpan model final...")
trainer.save_model(NEW_MODEL_PATH)
processor.save_pretrained(NEW_MODEL_PATH)

print(f"\nSelamat! Model baru Anda telah disimpan di folder '{NEW_MODEL_PATH}'")