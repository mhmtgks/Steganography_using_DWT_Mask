import cv2
import pywt
import numpy as np

def bits_to_text(bits):
    try:
        n = int(bits, 2)
        return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()
    except:
        return "[Hata: Mesaj çözülemedi]"


def extract_message_from_stego(
        stego_image_path,
        wavelet='haar',
        level=1,
        sub_band='HL'
    ):

    img = cv2.imread(stego_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Stego görüntü okunamadı")

    # DWT
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    target_idx = -level
    LH, HL, HH = coeffs[target_idx]

    if sub_band == 'LH':
        band = LH
    elif sub_band == 'HL':
        band = HL
    elif sub_band == 'HH':
        band = HH
    else:
        raise ValueError("Geçersiz bant")

    flat_band = np.abs(band).flatten()
    threshold = np.percentile(flat_band, 50)
    indices = np.where(flat_band > threshold)[0]

  
    h, w = img.shape
    bits = ""

    for idx in indices:
        i, j = divmod(idx, w)
        bits += str(img[i, j] & 1)

    delimiter = '1111111111111110'
    if delimiter in bits:
        return bits_to_text(bits.split(delimiter)[0])
    else:
        return "[Bitiş işareti bulunamadı]"

if __name__ == "__main__":
    stego_path = "C:/Users/mgoek/Desktop/Stego Images/47.png"
    
    message = extract_message_from_stego(
        stego_path,
        level=4,
        sub_band='LH'
    )
    
    print("Çözülen mesaj:", message)