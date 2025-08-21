# backend/validasi_konten.py
# Versi dengan metode pencarian "tanpa spasi" untuk mengatasi tokenization

import re
from typing import List, Dict, Any

def cek_kelengkapan_dokumen(laporan_kontekstual: List[Dict[str, Any]], aturan_kelengkapan: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Memvalidasi kelengkapan dokumen berdasarkan keberadaan frasa wajib
    menggunakan metode pencarian 'tanpa spasi' untuk mengatasi tokenization.
    """
    
    frasa_wajib = aturan_kelengkapan.get('frasa_wajib', [])
    if not frasa_wajib:
        return {"status": "DILEWATI", "message": "Tidak ada aturan frasa wajib yang didefinisikan."}

    # Gabungkan semua token dari semua halaman menjadi satu string teks besar
    teks_dokumen_lengkap = ""
    for halaman in laporan_kontekstual:
        analisis_halaman = halaman.get('analisis', {})
        hasil_analisis = analisis_halaman.get('hasil_analisis_kontekstual', [])
        for item in hasil_analisis:
            # Menggunakan 'token' yang sudah di-lowercase dari model jika ada, atau handle token dari tokenizer
            # Untuk LayoutLMV3, token biasanya sudah bersih
            token = item.get('token', '')
            # Menghilangkan karakter khusus awal kata dari beberapa tokenizer (seperti ' ')
            if token.startswith(' '):
                token = token[1:]
            teks_dokumen_lengkap += token

    # --- PERBAIKAN LOGIKA DI SINI ---
    # Normalisasi teks ke huruf kecil dan hapus SEMUA spasi dan karakter non-alfanumerik
    teks_dokumen_normal = re.sub(r'[^a-z0-9]', '', teks_dokumen_lengkap.lower())

    frasa_ditemukan = []
    frasa_tidak_ditemukan = []

    for frasa in frasa_wajib:
        # Normalisasi frasa yang dicari dengan cara yang sama
        frasa_normal = re.sub(r'[^a-z0-9]', '', frasa.lower())
        
        if frasa_normal in teks_dokumen_normal:
            frasa_ditemukan.append(frasa)
        else:
            frasa_tidak_ditemukan.append(frasa)

    status = "LENGKAP" if not frasa_tidak_ditemukan else "TIDAK LENGKAP"

    return {
        "status": status,
        "frasa_ditemukan": frasa_ditemukan,
        "frasa_tidak_ditemukan": frasa_tidak_ditemukan
    }