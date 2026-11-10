import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_auc_score
)

# 1. Memuat dataset publik
dataset = load_breast_cancer(as_frame=True)
X = dataset.data

# Pada dataset bawaan sklearn: 0 = malignant, 1 = benign.
# Target diubah agar kelas positif (1) adalah malignant.
y = (dataset.target == 0).astype(int)

print("Jumlah data:", len(X))
print("Jumlah fitur:", X.shape[1])
print("\nDistribusi kelas:")
print(y.map({0: "Benign", 1: "Malignant"}).value_counts())

# 2. Membagi data: 80% training dan 20% testing.
# stratify menjaga proporsi kelas pada kedua bagian.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Membuat dan melatih model Random Forest.
# Pembatasan kedalaman dan ukuran daun membantu mengurangi overfitting.
# class_weight='balanced' memberi bobot lebih besar pada kelas minoritas.
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# 4. Prediksi pada data testing yang tidak digunakan saat training.
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# 5. Evaluasi model.
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nAkurasi data training : {:.4f}".format(train_accuracy))
print("Akurasi data testing  : {:.4f}".format(test_accuracy))
print("ROC-AUC               : {:.4f}".format(
    roc_auc_score(y_test, y_probability)
))

print("\nConfusion Matrix")
print("Urutan label: Benign (0), Malignant (1)")
print(cm)

print("\nClassification Report")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    zero_division=0
))

# 6. Visualisasi confusion matrix.
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)
display.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix_hasil.png", dpi=180)
plt.show()

# 7. Visualisasi distribusi kelas.
class_counts = pd.Series(y).map({
    0: "Benign",
    1: "Malignant"
}).value_counts().reindex(["Benign", "Malignant"])

class_counts.plot(kind="bar", color=["steelblue", "salmon"])
plt.title("Distribusi Kelas Dataset")
plt.xlabel("Kelas")
plt.ylabel("Jumlah data")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("distribusi_kelas.png", dpi=180)
plt.show()
 