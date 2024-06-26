# 🔍 Otsu Thresholding with Python 📊

Bu Python projesi, histogram verilerini kullanarak bir görüntü matrisine Otsu eşikleme işlemi uygular. Bu işlem, optimal eşik değerini hesaplar, görüntü matrisine uygular ve sonuçları görselleştirir.

## 📄 İçindekiler
- [Giriş](#-giriş)
- [Gereksinimler](#-gereksinimler)
- [Kullanım](#-kullanım)
- [Örnek](#-örnek)
- [Açıklama](#-açıklama)
- [Sonuçlar](#-sonuçlar)


## 📘 Giriş
Otsu yöntemi, görüntü binarizasyonu için optimal eşik değerini hesaplayan bir tekniktir. Bu yöntem, iki sınıfın (arka plan ve ön plan) piksel yoğunluklarını en iyi şekilde ayıran eşik değerini bulur ve bu sınıfların iç varyansını minimize eder.

## 📦 Gereksinimler
- Python 3.x
- pandas
- numpy
- matplotlib

Gerekli paketleri yüklemek için:
```sh
pip install pandas numpy matplotlib
```

## 🚀 Kullanım

1. Depoyu klonlayın:
   ```sh
   git clone https://github.com/kullanici-adi/otsu-thresholding.git
   cd otsu-thresholding
   ```

2. Girdi Excel dosyanızı (`soru1_2_data.xlsx`) depo dizinine yerleştirin.

3. Script'i çalıştırın:
   ```sh
   python main.py
   ```

## 📝 Örnek

Script, yoğunluk değerlerinin histogramını ve piksel sayılarını kullanarak optimal eşik değerini hesaplar. Aşağıda bir örnek histogram verisi bulunmaktadır:

```python
histogram_data = {
    100: 12,101: 18,102: 32,103: 48,104: 52,105: 65,106: 55,107: 42,108: 32,109: 16,110: 10,
    140: 5,141: 18,142: 25,143: 32,144: 40,145: 65,146: 43,147: 32,148: 20,149: 10,150: 4
}
```

## 📖 Açıklama

Script şu adımları gerçekleştirir:
1. **Histogram Verisini Okuma**: Histogram verilerini dizilere dönüştürür.
2. **Piksel Yoğunluklarını Normalleştirme**: Olasılık Yoğunluk Fonksiyonu (PDF) ve Kümülatif Dağılım Fonksiyonu (CDF) hesaplar.
3. **Otsu Eşik Değeri Hesaplama**: Her olası eşik değeri için sınıf içi varyansı hesaplar ve bu varyansı minimize eden eşik değerini belirler.
4. **Görüntü Binarizasyonu**: Optimal eşik değerini görüntü matrisine uygular.
5. **Görselleştirme**: Orijinal gri tonlamalı görüntü ve binarize edilmiş görüntüyü gösterir. Ayrıca arka plan ve ön plan sınıflarının histogramlarını da görüntüler.

### Ana Hesaplamalar
- **Sınıf İçi Varyans**: \(\sigma_w^2 = \omega_1 \sigma_1^2 + \omega_2 \sigma_2^2\)
- **Arka Plan ve Ön Plan Ağırlıkları (\(\omega_1, \omega_2\))**:
  \[
  \omega_1 = \frac{\text{birinci sınıftaki piksel sayısı}}{\text{toplam piksel sayısı}}, \quad \omega_2 = \frac{\text{ikinci sınıftaki piksel sayısı}}{\text{toplam piksel sayısı}}
  \]
- **Arka Plan ve Ön Plan Ortalamaları (\(\mu_1, \mu_2\))**:
  \[
  \mu_1 = \frac{\sum (\text{yoğunluk değerleri} \times \text{birinci sınıftaki piksel sayıları})}{\text{birinci sınıftaki piksel sayısı}}, \quad \mu_2 = \frac{\sum (\text{yoğunluk değerleri} \times \text{ikinci sınıftaki piksel sayıları})}{\text{ikinci sınıftaki piksel sayısı}}
  \]

## 📊 Sonuçlar

Script'i çalıştırdıktan sonra optimal Otsu eşik değeri ve görüntülerin görselleştirmelerini elde edeceksiniz. Script ayrıca arka plan ve ön plan ağırlıkları, ortalamaları ve varyanslarını da yazdırır.

### Örnek Çıktı
```sh
Optimal Otsu Threshold: 110
Background Weight (Wb): 0.5650887573964497
Background Mean (Mb): 104.92931937172774
Background Variance (Vb): 5.589245086483375
Foreground Weight (Wf): 0.4349112426035503
Foreground Mean (Mf): 144.83333333333334
Foreground Variance (Vf): 4.791950113378685
```





