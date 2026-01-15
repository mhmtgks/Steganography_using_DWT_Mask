import numpy as np
import pywt
import cv2
import math
import pandas as pd


def text_to_bits(text):
    bits = bin(int.from_bytes(text.encode(), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))


def bits_to_text(bits):
    try:
        n = int(bits, 2)
        return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()
    except:
        return "[Hata: Mesaj çözülemedi]"


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ===================== EMBED (2 LSB) =====================
def embed_multi_level(image_path, secret_message, wavelet='haar', level=1, sub_band='HH'):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Görüntü okunamadı")

    coeffs = pywt.wavedec2(img, wavelet, level=level)
    LH, HL, HH = coeffs[-level]

    band = {'LH': LH, 'HL': HL, 'HH': HH}[sub_band]

    flat_band = np.abs(band).flatten()
    threshold = np.percentile(flat_band, 50)
    indices = np.where(flat_band > threshold)[0]

    bits = text_to_bits(secret_message) + '1111111111111110'
    if len(bits) % 2 != 0:
        bits += '0'

    if len(bits) > 2 * len(indices):
        raise ValueError("Mesaj kapasiteyi aşıyor")

    stego_img = img.copy()
    h, w = img.shape

    for k in range(0, len(bits), 2):
        idx = indices[k // 2]
        i, j = divmod(idx, w)
        two_bits = bits[k:k+2]
        stego_img[i, j] = (stego_img[i, j] & ~3) | int(two_bits, 2)

    return stego_img


# ===================== EXTRACT (2 LSB) =====================
def extract_multi_level(image_path, wavelet='haar', level=1, sub_band='HH'):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    coeffs = pywt.wavedec2(img, wavelet, level=level)
    LH, HL, HH = coeffs[-level]
    band = {'LH': LH, 'HL': HL, 'HH': HH}[sub_band]

    flat_band = np.abs(band).flatten()
    threshold = np.percentile(flat_band, 50)
    indices = np.where(flat_band > threshold)[0]

    bits = ""
    h, w = img.shape

    for idx in indices:
        i, j = divmod(idx, w)
        val = img[i, j] & 3
        bits += format(val, '02b')

    delimiter = '1111111111111110'
    if delimiter in bits:
        return bits_to_text(bits.split(delimiter)[0])

    return "Bitiş işareti bulunamadı"


# ===================== MAIN =====================
if __name__ == "__main__":

    image_path = 'C:/Users/mgoek/Desktop/Cover images'
    

    results = []
    msg = "It is always too mercy to anyone"

    SELECTED_LEVEL = 7
    
    for j in range(1,4):
        if j==1:
            SELECTED_BAND = 'LH'
        if j==2:
            SELECTED_BAND = 'HL'
        if j==3:
            SELECTED_BAND = 'HH'
        stego_path = f"C:/Users/mgoek/Desktop/Stego Images/Level-{SELECTED_LEVEL}-{SELECTED_BAND}-Low-Payload"
        for i in range(1, 86):
    
            full_path = f"{image_path}/{i}.jpg"
    
            stego_img = embed_multi_level(
                full_path,
                msg,
                level=SELECTED_LEVEL,
                sub_band=SELECTED_BAND
            )
    
            full_stego_path = f"{stego_path}/{i}.png"
            cv2.imwrite(full_stego_path, stego_img)
    
            decoded_msg = extract_multi_level(
                full_stego_path,
                level=SELECTED_LEVEL,
                sub_band=SELECTED_BAND
            )
    
            original_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            psnr_value = calculate_psnr(original_img, stego_img)
    
            print("-" * 30)
            print(i)
            print(f"[+] Çözülen Mesaj: {decoded_msg}")
            print(f"PSNR Değeri      : {psnr_value:.2f} dB")
    
            results.append({
                "Dosya İsmi": i,
                "PSNR Değeri (dB)": round(psnr_value, 2),
                "Mesaj": decoded_msg
            })

        df = pd.DataFrame(results)
        try:
            df = df.applymap(lambda x: x if all(ord(c) >= 32 for c in str(x)) else "It is always mercy")
            df.to_excel(f"{stego_path}/result.xlsx", index=False)
        except Exception as e:
            print(e)
