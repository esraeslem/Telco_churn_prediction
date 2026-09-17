##################################################
# TELCO CUSTOMER CHURN FEATURE ENGINEERING & CLASSIFICATION
##################################################

# İş Problemi
# -----------
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi
# modeli geliştirilmesi beklenmektedir. Modeli geliştirmeden önce gerekli
# olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz
# beklenmektedir.

# Veri Seti Hikayesi
# -------------------
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043
# müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom
# şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden
# ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.
# 21 Değişken, 7043 Gözlem

import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

RANDOM_STATE = 17


##################################################
# YARDIMCI FONKSİYONLAR
##################################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik ama kardinal
    değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        İsimleri alınmak istenen dataframe.
    cat_th : int, optional
        Numerik fakat kategorik olan değişkenler için sınıf eşik değeri.
    car_th : int, optional
        Kategorik fakat kardinal değişkenler için sınıf eşik değeri.

    Returns
    -------
    cat_cols : list
        Kategorik değişken listesi.
    num_cols : list
        Numerik değişken listesi.
    cat_but_car : list
        Kategorik görünümlü kardinal değişken listesi.
    """
    cat_cols = [
        col for col in dataframe.columns
        if str(dataframe[col].dtypes) in ["category", "object", "bool", "str"]
        or pd.api.types.is_string_dtype(dataframe[col])
    ]
    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int64", "float64"]
    ]
    cat_but_car = [
        col for col in cat_cols if dataframe[col].nunique() > car_th
    ]
    # NOT: set() kullanmıyoruz; Python'da hash randomization nedeniyle
    # set sırası çalıştırmalar arasında değişebilir ve bu da encoding
    # sonrası kolon sırasını, dolayısıyla model sonuçlarının (özellikle
    # ağaç tabanlı modellerde) yeniden üretilebilirliğini bozar.
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name):
    summary = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe),
    })
    print(summary)
    print("##########################################")


def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(
        pd.DataFrame({
            "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
            "Count": dataframe.groupby(categorical_col)[target].count(),
        })
    )
    print("##########################################")


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


##################################################
# GÖREV 1: KEŞİFÇİ VERİ ANALİZİ (EDA)
##################################################

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df.head()
df.shape
df.info()

# Adım 2: Gerekli düzenlemeleri yapınız (Tip hatası olan değişkenler gibi)
# TotalCharges sayısal bir değişken olmasına rağmen object/string olarak
# okunmuş (boşluk içeren 11 satır yüzünden). Sayısala çeviriyoruz.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Churn değişkenini sayısala çeviriyoruz (Yes: 1, No: 0)
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
print("cat_cols:", cat_cols)
print("num_cols:", num_cols)
print("cat_but_car:", cat_but_car)

# customerID kardinal bir değişken (her satırda eşsiz), modele katkısı yok.
# num_cols listesinde yer alan Churn'u analiz dışında tutuyoruz.
num_cols = [col for col in num_cols if col not in ["Churn"]]

# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col)

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# Adım 5: Aykırı gözlem var mı inceleyiniz.
for col in num_cols:
    print(col, check_outlier(df, col))

# Adım 6: Eksik gözlem var mı inceleyiniz.
missing_values_table(df)

##################################################
# Görsel çıktılar (README için)
##################################################

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df["Churn"].map({0: "Hayır", 1: "Evet"}).value_counts().plot(
    kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"]
)
axes[0].set_title("Churn Dağılımı")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", rotation=0)

sns.histplot(data=df, x="tenure", hue=df["Churn"].map({0: "Hayır", 1: "Evet"}), multiple="stack", ax=axes[1])
axes[1].set_title("Tenure (Ay) - Churn Dağılımı")
plt.tight_layout()
plt.savefig("outputs/churn_and_tenure_distribution.png", dpi=120)
plt.close()

plt.figure(figsize=(8, 6))
corr_df = df[num_cols + ["Churn"]].corr()
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Numerik Değişken Korelasyon Matrisi")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=120)
plt.close()


##################################################
# GÖREV 2: FEATURE ENGINEERING
##################################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# TotalCharges'daki eksik değerler tenure = 0 olan (o ay yeni kaydolmuş)
# 11 müşteriye ait olduğu için 0 ile dolduruyoruz.
# NOT: pandas 3.0'da Copy-on-Write varsayılan olduğundan
# df["col"].fillna(..., inplace=True) şeklindeki zincirleme atama orijinal
# dataframe'i güncellemez; bu yüzden doğrudan yeniden atama kullanılır.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Aykırı değer analizinde outlier bulunmadı (Adım 5), yine de olası
# üretim/yeni veri senaryoları için threshold fonksiyonu tanımlı bırakıldı.
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# Adım 2: Yeni değişkenler oluşturunuz.

# Tenure değişkeninden yıllık kategori oluşturma
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Yıl"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Yıl"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Yıl"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Yıl"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Yıl"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Yıl"

# 1 veya 2 yıllık sözleşmesi olan müşterileri "Engaged" (bağlı) olarak işaretleme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# Herhangi bir destek, yedekleme veya koruma hizmeti almayan müşteriler
df["NEW_noProt"] = df.apply(
    lambda x: 1 if (x["OnlineBackup"] != "Yes") and (x["DeviceProtection"] != "Yes") and (x["TechSupport"] != "Yes")
    else 0,
    axis=1,
)

# Aylık sözleşmesi olan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(
    lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1
)

# Kişinin toplam aldığı servis sayısı
services_cols = [
    "PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
df["NEW_TotalServices"] = (df[services_cols] == "Yes").sum(axis=1) + (
    df["InternetService"] != "No"
).astype(int)

# Herhangi bir streaming hizmeti alan kişi
df["NEW_FLAG_ANY_STREAMING"] = df.apply(
    lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1
)

# Kişi otomatik ödeme yapıyor mu
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(
    lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0
)

# Ortalama aylık ödeme (0'a bölme hatasını önlemek için tenure+1)
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Güncel fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df["NEW_TotalServices"] + 1)

df.head()
df.shape

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["Churn"]]

# Label Encoding (2 sınıflı değişkenler)
binary_cols = [
    col for col in df.columns
    if df[col].nunique() == 2 and (df[col].dtypes in ["object", "str"] or pd.api.types.is_string_dtype(df[col]))
]

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding (2'den fazla sınıflı kategorik değişkenler)
ohe_cols = [col for col in cat_cols if col not in binary_cols and col != "Churn"]
df = one_hot_encoder(df, ohe_cols, drop_first=True)

# customerID modele dahil edilmeyecek
df.drop("customerID", axis=1, inplace=True, errors="ignore")

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = RobustScaler()
final_num_cols = [col for col in num_cols if col in df.columns]
df[final_num_cols] = scaler.fit_transform(df[final_num_cols])

df.head()
df.shape

# bool tipindeki one-hot kolonlarını int'e çeviriyoruz (model uyumluluğu için)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)


##################################################
# GÖREV 3: MODELLEME
##################################################

y = df["Churn"]
X = df.drop(["Churn"], axis=1)

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy
# skorlarını inceleyip. En iyi 4 modeli seçiniz.

classifiers = [
    ("LR", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ("RF", RandomForestClassifier(random_state=RANDOM_STATE)),
    ("SVM", SVC(random_state=RANDOM_STATE)),
    ("Adaboost", AdaBoostClassifier(random_state=RANDOM_STATE)),
    ("GBM", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)),
    ("LightGBM", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)),
    ("CatBoost", CatBoostClassifier(verbose=False, random_state=RANDOM_STATE)),
]

base_results = []
print("\nBase Modeller (5-Fold CV):")
for name, classifier in classifiers:
    cv_results = cross_validate(classifier, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    acc = cv_results["test_accuracy"].mean()
    f1 = cv_results["test_f1"].mean()
    roc_auc = cv_results["test_roc_auc"].mean()
    base_results.append({"Model": name, "Accuracy": acc, "F1": f1, "ROC_AUC": roc_auc})
    print(f"{name}: Accuracy={round(acc, 4)}  F1={round(f1, 4)}  ROC_AUC={round(roc_auc, 4)}")

base_results_df = pd.DataFrame(base_results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\nAccuracy'ye göre sıralanmış base model sonuçları:")
print(base_results_df)

best_4_models = base_results_df.head(4)["Model"].tolist()
print("\nEn iyi 4 model:", best_4_models)

base_results_df.to_csv("outputs/base_model_results.csv", index=False)

# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu
# gerçekleştirin ve bulduğunuz hiperparametreler ile modeli tekrar
# kurunuz.

param_grids = {
    "LR": {"C": [0.01, 0.1, 1, 10]},
    "KNN": {"n_neighbors": range(2, 30, 2)},
    "CART": {"max_depth": range(1, 15), "min_samples_split": range(2, 20, 2)},
    "RF": {"max_depth": [5, 8, None], "n_estimators": [100, 200], "min_samples_split": [2, 8, 15]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Adaboost": {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.5, 1]},
    "GBM": {"learning_rate": [0.01, 0.1], "max_depth": [3, 5], "n_estimators": [100, 200]},
    "XGBoost": {"learning_rate": [0.01, 0.1], "max_depth": [3, 5], "n_estimators": [100, 200]},
    "LightGBM": {"learning_rate": [0.01, 0.1], "n_estimators": [100, 300]},
    "CatBoost": {"iterations": [200, 500], "learning_rate": [0.03, 0.1], "depth": [4, 6]},
}

model_lookup = dict(classifiers)

tuned_results = []
best_estimators = {}

print("\nHiperparametre Optimizasyonu (sadece en iyi 4 model için):")
for name in best_4_models:
    print(f"\n########## {name} ##########")
    base_model = model_lookup[name]
    params = param_grids[name]

    gs_best = GridSearchCV(base_model, params, cv=5, n_jobs=1, scoring="accuracy", verbose=0).fit(X, y)
    final_model = base_model.set_params(**gs_best.best_params_)

    cv_results = cross_validate(final_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    acc = cv_results["test_accuracy"].mean()
    f1 = cv_results["test_f1"].mean()
    roc_auc = cv_results["test_roc_auc"].mean()

    print(f"En iyi parametreler: {gs_best.best_params_}")
    print(f"Tuned Accuracy: {round(acc, 4)}  F1: {round(f1, 4)}  ROC_AUC: {round(roc_auc, 4)}")

    tuned_results.append({
        "Model": name,
        "Best_Params": gs_best.best_params_,
        "Accuracy": acc,
        "F1": f1,
        "ROC_AUC": roc_auc,
    })
    best_estimators[name] = final_model.fit(X, y)

tuned_results_df = pd.DataFrame(tuned_results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\nHiperparametre optimizasyonu sonrası sonuçlar:")
print(tuned_results_df[["Model", "Accuracy", "F1", "ROC_AUC"]])

tuned_results_df.to_csv("outputs/tuned_model_results.csv", index=False)

# En iyi modelin feature importance grafiği (varsa)
best_model_name = tuned_results_df.iloc[0]["Model"]
best_model = best_estimators[best_model_name]

if hasattr(best_model, "feature_importances_"):
    importances = pd.DataFrame(
        {"Feature": X.columns, "Importance": best_model.feature_importances_}
    ).sort_values("Importance", ascending=False).head(15)
    importance_title = f"{best_model_name} - Değişken Önem Düzeyleri (Top 15)"
elif hasattr(best_model, "coef_"):
    # Doğrusal modellerde (ör. Logistic Regression) katsayıların mutlak
    # değeri değişken önemi olarak kullanılır.
    importances = pd.DataFrame(
        {"Feature": X.columns, "Importance": np.abs(best_model.coef_[0])}
    ).sort_values("Importance", ascending=False).head(15)
    importance_title = f"{best_model_name} - |Katsayı| Değerlerine Göre Önem (Top 15)"
else:
    importances = None

if importances is not None:
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importances, x="Importance", y="Feature", color="#4C72B0")
    plt.title(importance_title)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=120)
    plt.close()

print("\nBitti. Sonuçlar 'outputs/' klasörüne kaydedildi.")
