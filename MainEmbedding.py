import numpy as np
import pywt
import cv2
import math
import os
import pandas as pd

def text_to_bits(text):
    bits = bin(int.from_bytes(text.encode(), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf') 
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def embed_multi_level(img, secret_message, wavelet='haar', level=1, sub_band='HH'):
    # img artık path değil, doğrudan numpy array olarak geliyor
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    target_idx = -level
    LH, HL, HH = coeffs[target_idx]

    if sub_band == 'LH': band = LH
    elif sub_band == 'HL': band = HL
    else: band = HH

    flat_band = np.abs(band).flatten()
    threshold = np.percentile(flat_band, 50)
    mask = flat_band > threshold
    indices = np.where(mask)[0]

    bits = text_to_bits(secret_message) + '1111111111111110'

    if len(bits) > len(indices):
        return None, "Kapasite yetersiz"

    stego_img = img.copy()
    h, w = img.shape

    for k, bit in enumerate(bits):
        idx = indices[k]
        i, j = divmod(idx, w)
        if i < h and j < w: # Sınır kontrolü
            stego_img[i, j] = (stego_img[i, j] & ~1) | int(bit)

    return stego_img, None

# --- ANA İŞLEM DÖNGÜSÜ ---

if __name__ == "__main__":
    # Klasör Yolları (Burayı kendi bilgisayarına göre güncelle)
    input_folder = 'C:/Users/mgoek/Desktop/Cover images/'
    output_folder = 'C:/Users/mgoek/Desktop/Stego Images/Level-1-HH-Low-Payload/'
    excel_path = 'C:/Users/mgoek/Desktop/Stego Images/Level-1-HH-Low-Payload/steganografi_analiz_HL1.xlsx'
    
    # Eğer çıktı klasörü yoksa oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    msg = "It is always too mercy to anyone"
    SELECTED_LEVEL = 1   
    SELECTED_BAND = 'HH'
    
    results = [] # Verileri toplamak için liste

    # Klasördeki dosyaları listele
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    print(f"{len(files)} adet resim bulundu. İşlem başlıyor...\n")

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        
        # Resmi oku (Gri tonlamalı)
        original_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if original_img is None:
            print(f"Hata: {filename} okunamadı, atlanıyor.")
            continue

        # Gömme işlemini yap
        stego_img, error = embed_multi_level(original_img, msg, level=SELECTED_LEVEL, sub_band=SELECTED_BAND)

        if error:
            print(f"Hata: {filename} için {error}")
            continue

        # Stego resmi kaydet
        base_name = os.path.splitext(filename)[0] # Dosya ismini al (örn: 'Im_33')
        output_filename = f"stego_{base_name}.png" # Uzantıyı .png olarak belirle
        output_path = os.path.join(output_folder, output_filename)
        
        # PNG olarak kaydet
        cv2.imwrite(output_path, stego_img)

        # PSNR Hesapla
        psnr_val = calculate_psnr(original_img, stego_img)
        
        # Listeye ekle
        results.append({
            "Dosya İsmi": filename,
            "PSNR Değeri (dB)": round(psnr_val, 2)
        })
        
        print(f"İşlendi: {filename} -> PSNR: {psnr_val:.2f} dB")

    # --- EXCEL'E AKTARMA ---
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)

    print("-" * 30)
    print(f"İşlem Tamamlandı!")
    print(f"Toplam {len(results)} resim işlendi.")
    print(f"Sonuçlar kaydedildi: {excel_path}")