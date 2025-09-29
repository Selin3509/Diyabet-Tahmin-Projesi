import joblib
import numpy as np

# 📌 1️⃣ Modeli yükle
model = joblib.load("gradient_boosting_model.pkl")

# 📌 2️⃣ Örnek veri ile tahmin yap
sample_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # Örnek hasta

# 📌 3️⃣ Tahmin yap
prediction = model.predict(sample_data)

# 📌 4️⃣ Sonuçları yazdır
result = "Diyabetli" if prediction[0] == 1 else "Diyabetli Değil"
print("Tahmin Sonucu:", result)

import joblib
import numpy as np

# 📌 1️⃣ Modeli yükle
model = joblib.load("gradient_boosting_model.pkl")

# 📌 2️⃣ Kullanıcıdan verileri al
print("Lütfen aşağıdaki verileri giriniz:")

# Verileri kullanıcıdan al
pregnancies = int(input("Hamilelik sayısı (Pregnancies): "))
glucose = int(input("Glukoz (Glucose): "))
blood_pressure = int(input("Kan basıncı (Blood Pressure): "))
skin_thickness = int(input("Cilt kalınlığı (Skin Thickness): "))
insulin = int(input("İnsülin (Insulin): "))
bmi = float(input("Vücut kitle indeksi (BMI): "))
diabetes_pedigree = float(input("Diyabet soyağacı (Diabetes Pedigree Function): "))
age = int(input("Yaş (Age): "))

# 📌 3️⃣ Kullanıcıdan alınan verileri numpy array formatına çevir
user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

# 📌 4️⃣ Tahmin yap
prediction = model.predict(user_data)

# 📌 5️⃣ Sonucu yazdır
result = "Diyabetli" if prediction[0] == 1 else "Diyabetli Değil"
print(f"Tahmin Sonucu: {result}")
