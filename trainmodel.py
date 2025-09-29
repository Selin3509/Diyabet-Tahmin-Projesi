import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# 📌 1️⃣ Veriyi Yükle
df = pd.read_csv("archive.zip")

# Bağımsız (özellikler) ve bağımlı (hedef) değişkenleri ayır
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# 📌 2️⃣ Eğitim ve Test Verisine Ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# 📌 3️⃣ Veriyi Dengeleme (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 📌 4️⃣ Modeli Eğit
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_resampled, y_resampled)

# 📌 5️⃣ Modeli Test Et
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Yeni Model Doğruluk Oranı: {accuracy:.2f}")

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Modeli tanımla
model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)

# Modeli eğit
model.fit(X_resampled, y_resampled)

# Modeli test et
y_pred = model.predict(X_test)

# Sonuçları yazdır
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Doğruluk Oranı: {accuracy:.2f}")

from sklearn.svm import SVC  # SVC modelini içeri aktar

# 📌 SVM Modelini Tanımla
svm_model = SVC(kernel="linear", C=1.0, random_state=42)

# 📌 Modeli Eğit
svm_model.fit(X_resampled, y_resampled)

# 📌 Modeli Test Et
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 📌 Sonuçları Yazdır
print(f"SVM Model Doğruluk Oranı: {accuracy:.2f}")

from sklearn.neighbors import KNeighborsClassifier

# Modeli tanımla ve eğit
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_resampled, y_resampled)

# Modeli test et
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Sonuçları yazdır
print(f"📌 KNN Model Doğruluk Oranı: {accuracy_knn:.2f}")




from sklearn.ensemble import GradientBoostingClassifier

# Modeli tanımla ve eğit
gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_resampled, y_resampled)

# Modeli test et
y_pred_gb = gb_model.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

# Sonuçları yazdır
print(f"📌 Gradient Boosting Model Doğruluk Oranı: {accuracy_gb:.2f}")

import joblib  # joblib kütüphanesini ekle
# Modeli kaydet
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
print("Model başarıyla kaydedildi!")



