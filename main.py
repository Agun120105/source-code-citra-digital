"""HelloImageProject - Contoh manipulasi citra digital (bilateral denoise + unsharp mask)
File ini memberikan implementasi sederhana yang menjaga tampilan foto tetap photorealistic.
Metode:
  1) Bilateral filter untuk mereduksi noise sambil mempertahankan tepi.
  2) Unsharp mask (sharpening) ringan untuk mempertajam detail tanpa membuatnya terlihat 'digital'.

Petunjuk singkat:
  - Letakkan gambar di folder 'images' (sudah disediakan; file default: img-3.jpg)
  - Jalankan: python main.py
  - Output disimpan di folder 'output' sebagai 'result_<nama_gambar>'
"""

import os
import cv2
import numpy as np

# --- Pengaturan file / folder ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nama gambar default (diambil dari paket proyek)
INPUT_IMAGE_NAME = 'img-3.jpg'
INPUT_PATH = os.path.join(IMAGES_DIR, INPUT_IMAGE_NAME)

def bilateral_denoise(img, d=9, sigma_color=75, sigma_space=75):
    """Lakukan bilateral filter pada citra warna.
    Parameter:
      - d: diameter pixel yang dipakai (int). Nilai kecil untuk efek lebih ringan.
      - sigma_color: semakin besar, semakin banyak perbedaan warna yang disamakan.
      - sigma_space: seberapa jauh pengaruh filter (spasial).
    Mengapa bilateral? Karena ia mereduksi noise sambil mempertahankan tepi (important agar tetap terlihat foto).
    """
    # OpenCV expects BGR, img disini diasumsikan BGR (hasil cv2.imread)
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def unsharp_mask(img, amount=1.0, radius=5):
    """Unsharp mask sederhana:
    1. Blur citra
    2. Buat mask = original - blurred
    3. hasil = original + amount * mask
    Parameter:
      - amount: skala penguatan detail; 0.5-1.5 biasa aman
      - radius: sigma / kernel size untuk Gaussian blur (int)
    Pastikan hasil tetap dalam rentang [0,255] dan bertipe uint8.
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=radius, sigmaY=radius)
    # Convert ke float untuk operasi
    img_f = img.astype(np.float32)
    blurred_f = blurred.astype(np.float32)
    mask = img_f - blurred_f
    result = img_f + amount * mask
    # Clip dan kembali ke uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def process_image(path_in, path_out):
    """Proses end-to-end:
    - baca file
    - bilateral denoise
    - unsharp mask (ringan)
    - simpan hasil
    """
    if not os.path.isfile(path_in):
        raise FileNotFoundError(f"File input tidak ditemukan: {path_in}")
    # Baca citra sebagai warna (BGR)
    img = cv2.imread(path_in, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Gagal membaca citra — format mungkin tidak didukung atau file korup.')

    # Tampilkan ukuran dan tipe (untuk debugging / penjelasan)
    h, w = img.shape[:2]
    print(f"Membaca '{path_in}' — resolusi: {w}x{h}, dtype: {img.dtype}")

    # 1) Bilateral filter -> mereduksi noise namun mempertahankan tepi
    denoised = bilateral_denoise(img, d=9, sigma_color=75, sigma_space=75)

    # 2) Unsharp mask -> menambah ketajaman detail secara halus
    sharpened = unsharp_mask(denoised, amount=1.0, radius=3)

    # Simpan hasil
    base = os.path.splitext(os.path.basename(path_in))[0]
    out_name = f"result_{base}.png"
    out_path = os.path.join(path_out, out_name)
    cv2.imwrite(out_path, sharpened)
    print(f"Selesai. Hasil disimpan di: {out_path}")

if __name__ == '__main__':
    print('--- HelloImageProject - proses mulai ---')
    process_image(INPUT_PATH, OUTPUT_DIR)
