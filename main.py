import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Histogram verilerini tanımlayalım
histogram_data = {
    100: 12,101: 18,102: 32,103: 48,104: 52,105: 65,106: 55,107: 42,108: 32,109: 16,
    110: 10,140: 5,141: 18,142: 25,143: 32,144: 40,145: 65,146: 43,147: 32,148: 20,149: 10,150: 4
}

# Histogram verilerini array'e dönüştürme
intensity_values = np.array(list(histogram_data.keys()))
pixel_counts = np.array(list(histogram_data.values()))

# Toplam piksel sayısı
total_pixels = np.sum(pixel_counts)

# Piksel yoğunluklarını normalize etme (PDF)
pdf = pixel_counts / total_pixels

# Kümülatif dağılım fonksiyonu (CDF)
cdf = np.cumsum(pdf)

# Otsu eşikleme için hesaplamalar
within_class_variance = np.zeros_like(intensity_values, dtype=float)
for i, t in enumerate(intensity_values):
    # Background (sınıf 1) ve foreground (sınıf 2) olasılıkları
    p1 = cdf[i]
    p2 = 1 - p1

    # Background ve foreground ortalamaları
    mu1 = np.sum(intensity_values[:i + 1] * pdf[:i + 1]) / p1 if p1 > 0 else 0
    mu2 = np.sum(intensity_values[i + 1:] * pdf[i + 1:]) / p2 if p2 > 0 else 0

    # Background ve foreground varyansları
    var1 = np.sum(((intensity_values[:i + 1] - mu1) ** 2) * pdf[:i + 1]) / p1 if p1 > 0 else 0
    var2 = np.sum(((intensity_values[i + 1:] - mu2) ** 2) * pdf[i + 1:]) / p2 if p2 > 0 else 0

    # Sınıf içi varyans
    within_class_variance[i] = p1 * var1 + p2 * var2

# En düşük sınıf içi varyansa sahip eşik değeri bulalım
optimal_threshold = intensity_values[np.argmin(within_class_variance)]
print(f"Optimal Otsu Threshold: {optimal_threshold}")

# Excel dosyasını okuyalım
file_path = 'matris_files/soru1_2_data.xlsx'
data = pd.read_excel(file_path, header=None)

# Veriyi 2D numpy array'e dönüştürelim
matrix = data.values

# Otsu eşik değerini matrise uygulayalım
binary_matrix = (matrix >= optimal_threshold).astype(int)

# Orijinal ve binary görüntüleri gösterelim
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(matrix, cmap='gray')
ax[0].set_title('Original Gray Scale Image')
ax[1].imshow(binary_matrix, cmap='gray')
ax[1].set_title('Binary Image after Otsu Thresholding')
plt.show()

# Background ve foreground için hesaplamalar
threshold = optimal_threshold
background_pixels = pixel_counts[intensity_values <= threshold]
foreground_pixels = pixel_counts[intensity_values > threshold]

# Background için hesaplamalar
wb = np.sum(background_pixels) / total_pixels
mb = np.sum(intensity_values[intensity_values <= threshold] * background_pixels) / np.sum(background_pixels)
vb = np.sum(((intensity_values[intensity_values <= threshold] - mb) ** 2) * background_pixels) / np.sum(
    background_pixels)

# Foreground için hesaplamalar
wf = np.sum(foreground_pixels) / total_pixels
mf = np.sum(intensity_values[intensity_values > threshold] * foreground_pixels) / np.sum(foreground_pixels)
vf = np.sum(((intensity_values[intensity_values > threshold] - mf) ** 2) * foreground_pixels) / np.sum(
    foreground_pixels)

print(f"Background Weight (Wb): {wb}")
print(f"Background Mean (Mb): {mb}")
print(f"Background Variance (Vb): {vb}")
print(f"Foreground Weight (Wf): {wf}")
print(f"Foreground Mean (Mf): {mf}")
print(f"Foreground Variance (Vf): {vf}")

# Hesaplama sonuçlarını ve histogramları gösterelim
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].bar(intensity_values, pixel_counts)
axs[0, 0].set_title('Histogram')
axs[0, 1].imshow(matrix, cmap='gray')
axs[0, 1].set_title('Original Gray Scale Image')
axs[1, 0].bar(intensity_values[intensity_values <= threshold], background_pixels)
axs[1, 0].set_title('Background Histogram')
axs[1, 1].bar(intensity_values[intensity_values > threshold], foreground_pixels)
axs[1, 1].set_title('Foreground Histogram')
plt.show()

# Sınıf içi varyansların grafiği
plt.figure(figsize=(10, 6))
plt.plot(intensity_values, within_class_variance, label='Within Class Variance')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold}')
plt.xlabel('Intensity Value')
plt.ylabel('Within Class Variance')
plt.title('Within Class Variance for Each Threshold')
plt.legend()
plt.show()
