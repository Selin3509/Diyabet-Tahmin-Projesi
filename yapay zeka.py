import pandas as pd

# Veriyi yükle
df = pd.read_csv("archive.zip")

# İlk 5 satırı göster
print(df.head())

# Eksik verileri kontrol et
print(df.isnull().sum())


