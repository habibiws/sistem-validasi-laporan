# backend/validasi_konten.py

import re
from typing import List, Dict, Any

def cek_kelengkapan_dokumen(laporan_kontekstual: List[Dict[str, Any]], aturan_kelengkapan: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Memvalidasi kelengkapan dokumen berdasarkan keberadaan frasa wajib.

    Args:
        laporan_kontekstual: Data hasil analisis AI dari `laporan_kontekstual.json`.
        aturan_kelengkapan: Dictionary berisi aturan, contoh: {'frasa_wajib': ['BERITA ACARA', 'TELKOM AKSES']}

    Returns:
        Sebuah dictionary berisi laporan hasil validasi kelengkapan.
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
            teks_dokumen_lengkap += item.get('token', '') + " "
    
    # Normalisasi teks ke huruf kecil untuk pencarian case-insensitive
    teks_dokumen_lengkap = teks_dokumen_lengkap.lower()
    # Hapus spasi ganda untuk memastikan pencarian frasa bekerja dengan benar
    teks_dokumen_lengkap = re.sub(r'\s+', ' ', teks_dokumen_lengkap)

    frasa_ditemukan = []
    frasa_tidak_ditemukan = []

    for frasa in frasa_wajib:
        # Normalisasi frasa yang dicari ke huruf kecil
        if frasa.lower() in teks_dokumen_lengkap:
            frasa_ditemukan.append(frasa)
        else:
            frasa_tidak_ditemukan.append(frasa)

    status = "LENGKAP" if not frasa_tidak_ditemukan else "TIDAK LENGKAP"

    return {
        "status": status,
        "frasa_ditemukan": frasa_ditemukan,
        "frasa_tidak_ditemukan": frasa_tidak_ditemukan
    }