import numpy as np
import pywt
import cv2
import pandas as pd
import os

def text_to_bits(text):
    bits = bin(int.from_bytes(text.encode(), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def embed_and_calculate_capacity(image_path, secret_message, wavelet='haar', level=1, sub_band='HH'):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")

    # Wavelet dönüşümü
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    
    try:
        LH, HL, HH = coeffs[-level]
    except IndexError:
        raise ValueError("Seçilen seviye (level) görüntü boyutu için çok büyük.")

    band = {'LH': LH, 'HL': HL, 'HH': HH}[sub_band]

    flat_band = np.abs(band).flatten()
    threshold = np.percentile(flat_band, 50) 
    indices = np.where(flat_band > threshold)[0]


    pixel_sayisi = len(indices)
    max_bits = pixel_sayisi * 2
    max_chars = max_bits // 8  

    # --- GÖMME İŞLEMİ ---
    bits = text_to_bits(secret_message) + '1111111111111110' 
    if len(bits) % 2 != 0:
        bits += '0'


    if len(bits) > max_bits:
        print(f"UYARI: Mesaj boyutu ({len(bits)} bit), kapasiteyi ({max_bits} bit) aşıyor! Mesaj kesilecek.")
        bits = bits[:max_bits] # Sığdığı kadarını al

    stego_img = img.copy()
    h, w = img.shape

    for k in range(0, len(bits), 2):
        idx = indices[k // 2]
        i, j = divmod(idx, w) 
        
       
        if i < h and j < w:
            two_bits = bits[k:k+2]
            stego_img[i, j] = (stego_img[i, j] & ~3) | int(two_bits, 2)

    return stego_img, max_chars

# ===================== MAIN =====================
if __name__ == "__main__":

    image_path = 'C:/Users/mgoek/Desktop/Cover images'
    
    msg = "It is always too mercy to anyone"
    SELECTED_LEVEL = 7 
    
    bands = ['LH', 'HL', 'HH']

    for band_name in bands:
        SELECTED_BAND = band_name
        
        # Klasör yoksa oluştur
        stego_path = f"C:/Users/mgoek/Desktop/Stego Images/Level-{SELECTED_LEVEL}-{SELECTED_BAND}-Low-Payload"
        os.makedirs(stego_path, exist_ok=True)
        
        results = []

        print(f"\n--- İşleniyor: Band {SELECTED_BAND} ---\n")

        for i in range(1, 86): 
            full_path = f"{image_path}/{i}.jpg"
            
            if not os.path.exists(full_path):
                print(f"Dosya bulunamadı atlanıyor: {full_path}")
                continue

            try:
                stego_img, capacity = embed_and_calculate_capacity(
                    full_path,
                    msg,
                    level=SELECTED_LEVEL,
                    sub_band=SELECTED_BAND
                )

                full_stego_path = f"{stego_path}/{i}.png"
                cv2.imwrite(full_stego_path, stego_img)

                print(f"Resim: {i}.jpg | Kapasite: {capacity} karakter")

                results.append({
                    "Dosya İsmi": i,
                    "Kapasite (Karakter)": capacity,
                    "Bant": SELECTED_BAND,
                    "Seviye": SELECTED_LEVEL
                })
            
            except Exception as e:
                print(f"Hata oluştu ({i}.jpg): {e}")

        df = pd.DataFrame(results)
        excel_save_path = f"{stego_path}/result_capacity.xlsx"
        df.to_excel(excel_save_path, index=False)
        print(f"Excel dosyası oluşturuldu: {excel_save_path}")