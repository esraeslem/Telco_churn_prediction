# 📞 Telco Customer Churn Prediction

Bu proje, **Miuul Data Scientist Bootcamp** kapsamında verilen "Telco Churn Prediction" vaka çalışmasının uçtan uca çözümüdür. Bir telekom şirketinin müşteri verileri üzerinde keşifçi veri analizi (EDA), özellik mühendisliği (Feature Engineering) ve sınıflandırma modelleri kurularak **müşteri kaybı (churn) tahmini** yapılmıştır.

## 📌 İş Problemi

Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarının gerçekleştirilmesi gerekmektedir.

## 🗂️ Veri Seti Hikayesi

Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki **7.043 müşteriye** ev telefonu ve internet hizmeti sağlayan hayali bir telekom şirketine aittir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

**21 Değişken, 7.043 Gözlem**

| Değişken | Açıklama |
|---|---|
| CustomerId | Müşteri İd'si |
| Gender | Cinsiyet |
| SeniorCitizen | Müşterinin yaşlı olup olmadığı (1, 0) |
| Partner | Müşterinin bir ortağı olup olmadığı (Evet, Hayır) |
| Dependents | Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı |
| tenure | Müşterinin şirkette kaldığı ay sayısı |
| PhoneService | Müşterinin telefon hizmeti olup olmadığı |
| MultipleLines | Müşterinin birden fazla hattı olup olmadığı |
| InternetService | Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır) |
| OnlineSecurity | Müşterinin çevrimiçi güvenliğinin olup olmadığı |
| OnlineBackup | Müşterinin online yedeğinin olup olmadığı |
| DeviceProtection | Müşterinin cihaz korumasına sahip olup olmadığı |
| TechSupport | Müşterinin teknik destek alıp almadığı |
| StreamingTV | Müşterinin TV yayını olup olmadığı |
| StreamingMovies | Müşterinin film akışı olup olmadığı |
| Contract | Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl) |
| PaperlessBilling | Müşterinin kağıtsız faturası olup olmadığı |
| PaymentMethod | Müşterinin ödeme yöntemi |
| MonthlyCharges | Müşteriden aylık olarak tahsil edilen tutar |
| TotalCharges | Müşteriden tahsil edilen toplam tutar |
| **Churn** | **Müşterinin kaybedilip kaybedilmediği (hedef değişken)** |

## 🧭 Metodoloji

### Görev 1: Keşifçi Veri Analizi (EDA)
- Numerik ve kategorik değişkenlerin yakalanması (`grab_col_names`)
- Tip hatalarının düzeltilmesi (`TotalCharges` object → numerik)
- Numerik/kategorik değişkenlerin dağılımlarının incelenmesi
- Kategorik değişkenler ile hedef değişken (`Churn`) ilişkisinin incelenmesi
- Aykırı gözlem analizi (IQR yöntemi)
- Eksik gözlem analizi

### Görev 2: Feature Engineering
- Eksik değerlerin (11 adet `TotalCharges`, hepsi `tenure=0` yeni müşteri) doldurulması
- **9 yeni değişken** türetilmesi (aşağıda listelenmiştir)
- Encoding: Label Encoding (ikili değişkenler) + One-Hot Encoding (çok sınıflılar)
- `RobustScaler` ile numerik değişkenlerin standartlaştırılması

**Türetilen değişkenler:**
| Değişken | Açıklama |
|---|---|
| `NEW_TENURE_YEAR` | Tenure'ı yıllık dilimlere ayırır (0-1, 1-2, ... 5-6 yıl) |
| `NEW_Engaged` | 1 veya 2 yıllık sözleşmesi olan müşteri mi |
| `NEW_noProt` | Yedekleme, cihaz koruması veya teknik destek almayan müşteri mi |
| `NEW_Young_Not_Engaged` | Sözleşmesi olmayan genç (senior olmayan) müşteri mi |
| `NEW_TotalServices` | Müşterinin aldığı toplam hizmet sayısı |
| `NEW_FLAG_ANY_STREAMING` | TV veya film akışı hizmeti alıyor mu |
| `NEW_FLAG_AutoPayment` | Otomatik ödeme yöntemi kullanıyor mu |
| `NEW_AVG_Charges` | Ortalama aylık harcama (`TotalCharges / (tenure+1)`) |
| `NEW_Increase` | Güncel faturanın ortalamaya göre artışı |
| `NEW_AVG_Service_Fee` | Hizmet başına düşen ücret |

### Görev 3: Modelleme
- 10 farklı sınıflandırma algoritması ile **5-fold cross validation** kullanılarak base model kurulumu (Accuracy, F1, ROC-AUC)
- Accuracy'ye göre en iyi **4 model** seçilmesi
- `GridSearchCV` ile hiperparametre optimizasyonu ve final modellerin yeniden kurulması

## 📊 Sonuçlar

### Base Model Karşılaştırması (5-Fold CV)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **LR** | **0.8065** | **0.5981** | **0.8489** |
| GBM | 0.8019 | 0.5798 | 0.8441 |
| Adaboost | 0.8008 | 0.5773 | 0.8431 |
| SVM | 0.7995 | 0.5726 | 0.8017 |
| CatBoost | 0.7984 | 0.5764 | 0.8405 |
| LightGBM | 0.7937 | 0.5715 | 0.8344 |
| RF | 0.7901 | 0.5538 | 0.8262 |
| XGBoost | 0.7835 | 0.5571 | 0.8215 |
| KNN | 0.7708 | 0.5488 | 0.7825 |
| CART | 0.7315 | 0.5034 | 0.6625 |

### Hiperparametre Optimizasyonu Sonrası (En İyi 4 Model)

| Model | En İyi Parametreler | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| **LR** | `C=1` | **0.8065** | **0.5981** | **0.8489** |
| GBM | `learning_rate=0.1, max_depth=3, n_estimators=100` | 0.8019 | 0.5798 | 0.8441 |
| Adaboost | `learning_rate=1, n_estimators=200` | 0.8012 | 0.5802 | 0.8441 |
| SVM | `C=1, kernel=rbf` | 0.7995 | 0.5726 | 0.8017 |

**En iyi model: Logistic Regression** — %80.65 accuracy ve 0.849 ROC-AUC ile en yüksek performansı göstermiştir. Basit doğrusal modelin ağaç tabanlı/ensemble modelleri geçmesi, özellik mühendisliği adımlarında oluşturulan değişkenlerin (özellikle `NEW_Engaged`, `tenure`, sözleşme tipi) hedef değişkenle güçlü ve büyük ölçüde doğrusal bir ilişki taşıdığına işaret etmektedir.

### Öne Çıkan Bulgular
- Churn oranı **%26.5** (1.869 / 7.043 müşteri) — sınıflar dengesiz.
- `tenure` ve `Contract` (sözleşme tipi), churn ile en güçlü ilişkiye sahip değişkenler: **aydan aya sözleşmesi olan** müşterilerde churn oranı **%42.7** iken, **2 yıllık sözleşmesi olanlarda %2.8**'e düşüyor.
- `Fiber optic` internet hizmeti alan müşterilerde churn oranı (%41.9), `DSL` kullananlara (%19.0) göre belirgin şekilde yüksek.
- Modelin en önemli değişkenleri: `tenure`, `InternetService_Fiber optic`, `Contract_Two year`, `NEW_Engaged`.

## 📈 Görselleştirmeler

`outputs/` klasöründe üretilir:
- `churn_and_tenure_distribution.png` — Churn dağılımı ve tenure'a göre churn kırılımı
- `correlation_heatmap.png` — Numerik değişkenler arası korelasyon matrisi
- `feature_importance.png` — En iyi modelin değişken önem düzeyleri
- `base_model_results.csv`, `tuned_model_results.csv` — Model karşılaştırma tabloları

## 📁 Proje Yapısı

```
telco-churn-prediction/
├── data/
│   └── Telco-Customer-Churn.csv
├── outputs/
│   ├── base_model_results.csv
│   ├── tuned_model_results.csv
│   ├── churn_and_tenure_distribution.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
├── telco_churn_prediction.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ Kurulum ve Çalıştırma

```bash
# Depoyu klonlayın
git clone https://github.com/<kullanici-adiniz>/telco-churn-prediction.git
cd telco-churn-prediction

# Sanal ortam oluşturup bağımlılıkları kurun
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Script'i çalıştırın
python telco_churn_prediction.py
```

> Not: `pandas>=3.0` sürümünde string sütunlar artık `object` yerine yerel `string` dtype ile okunur. Bu nedenle `grab_col_names` fonksiyonu `pd.api.types.is_string_dtype` kontrolü kullanır; eski `dtype == "O"` kontrolü bu sürümde çalışmaz.

## 🛠️ Kullanılan Teknolojiler

- Python 3
- pandas, numpy
- scikit-learn (LogisticRegression, KNN, CART, RandomForest, SVM, AdaBoost, GBM)
- XGBoost, LightGBM, CatBoost
- matplotlib, seaborn

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---
*Bu proje [Miuul](https://www.miuul.com/) Data Scientist Bootcamp programı kapsamında hazırlanmıştır.*
