import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. MEMBUAT DUMMY DATASET (RAW DATA)
np.random.seed(42)
data = {
    'waktu_berangkat': pd.date_range(start='2026-09-01 06:00:00', periods=100, freq='5h'),
    'jarak_ke_kampus_km': np.random.uniform(2.0, 15.0, 100), 
    'kendaraan': np.random.choice(['Motor', 'Mobil', 'Ojek Online'], 100),
    'cuaca': np.random.choice(['Cerah', 'Mendung', 'Gerimis', 'Hujan Deras'], 100)
}
df = pd.DataFrame(data)

# Simulasi label target (waktu tempuh aktual)
# Asumsi: Jarak jauh, cuaca buruk -> waktu tempuh lebih lama
base_time = df['jarak_ke_kampus_km'] * 4
weather_penalty = df['cuaca'].map({'Cerah': 0, 'Mendung': 5, 'Gerimis': 10, 'Hujan Deras': 20})
df['waktu_tempuh_menit'] = base_time + weather_penalty + np.random.normal(0, 3, 100)

print("=== 1. Data Mentah (Raw Data) ===")
print(df[['waktu_berangkat', 'kendaraan', 'cuaca']].head(), "\n")

# 2. PROSES FEATURE ENGINEERING
df_engineered = df.copy()

# a. Ekstraksi Fitur Waktu (Datetime)
df_engineered['jam'] = df_engineered['waktu_berangkat'].dt.hour
df_engineered['is_weekend'] = df_engineered['waktu_berangkat'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
# Jam sibuk: 06:00-09:00 atau 16:00-18:00
df_engineered['is_jam_sibuk'] = df_engineered['jam'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 18) else 0)

# b. Ordinal Encoding (Fitur Cuaca bertingkat)
cuaca_mapping = {'Cerah': 0, 'Mendung': 1, 'Gerimis': 2, 'Hujan Deras': 3}
df_engineered['cuaca_encoded'] = df_engineered['cuaca'].map(cuaca_mapping)

# c. One-Hot Encoding (Fitur Kendaraan)
df_engineered = pd.get_dummies(df_engineered, columns=['kendaraan'], drop_first=True)

# Drop kolom teks mentah yang sudah diekstrak
df_engineered.drop(columns=['waktu_berangkat', 'cuaca', 'jam'], inplace=True)

print("=== 2. Data Setelah Feature Engineering ===")
print(df_engineered.head(), "\n")

# 3. EVALUASI MODEL
# Memisahkan fitur (X) dan target (y)
X = df_engineered.drop(columns=['waktu_tempuh_menit'])
y = df_engineered['waktu_tempuh_menit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model dengan Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
prediksi = model.predict(X_test)

# Menghitung Error (MAE)
mae = mean_absolute_error(y_test, prediksi)
print("=== 3. Hasil Evaluasi ===")
print(f"Mean Absolute Error (MAE): {mae:.2f} menit")
print(f"Artinya: Rata-rata prediksi model meleset sekitar {mae:.2f} menit dari waktu aslinya.")