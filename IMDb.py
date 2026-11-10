import pandas as pd

# 1. Memuat Dataset
url = 'https://raw.githubusercontent.com/jigsawlabs-student/decision-trees-intro/master/3-practical-ds-4/imdb_movies.csv'
movies_df = pd.read_csv(url)

# Menampilkan 5 baris pertama dari dataset awal (seperti instruksi [1])
print("=== 5 Baris Pertama movies_df ===")
print(movies_df.head(), "\n")

# 2. Menyeleksi Fitur (X)
# Memilih kolom genre, budget, runtime, year, dan month
feature_columns = ['genre', 'budget', 'runtime', 'year', 'month']
X = movies_df[feature_columns]

# 3. Menyeleksi Target (y)
# Memilih revenue sebagai variabel yang akan diprediksi
y = movies_df['revenue']

# 4. Memeriksa Variabel X dan y (seperti instruksi [4] dan [5])
print("=== Pengecekan Variabel Target (y) - 2 data pertama ===")
print(y[:2], "\n")

print("=== Pengecekan Variabel Fitur (X) - 2 data pertama ===")
print(X[:2])