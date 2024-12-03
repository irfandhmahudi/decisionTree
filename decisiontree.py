import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

# Membaca dataset
data = pd.read_csv('Processed_Dataset_Matanajwa.csv', delimiter=',', header=0)

# Memastikan kolom 'Sentimen' ada dan diubah menjadi integer
data["Sentimen"] = pd.factorize(data["Sentimen"])[0]

# Menghapus kolom 'No' jika ada
if 'No' in data.columns:
    data = data.drop(labels="No", axis=1)

# Identifikasi dan transformasi kolom non-numerik
non_numeric_columns = data.select_dtypes(include=['object']).columns
for col in non_numeric_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Verifikasi dataset
print(data.head())
print(data.info())

# Mengubah dataset menjadi array numpy
data = data.to_numpy()

# Membagi dataset menjadi atribut dan label
inputData = data[:, :-1]  # Semua kolom kecuali terakhir
labelData = data[:, -1]   # Kolom terakhir sebagai label

# Menggunakan data yang sama untuk training dan testing
inputTraining = inputData
labelTraining = labelData
inputTesting = inputData
labelTesting = labelData

# Mendefinisikan decision tree classifier
model = tree.DecisionTreeClassifier()

# Melatih model
model = model.fit(inputTraining, labelTraining)

# Memprediksi label
hasilPrediksi = model.predict(inputTesting)
print("Hasil prediksi: ", hasilPrediksi)
print("Label sebenarnya: ", labelTesting)

# Menghitung akurasi
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("Prediksi benar: ", prediksiBenar, "data")
print("Prediksi salah: ", prediksiSalah, "data")
print("Akurasi: ", prediksiBenar / (prediksiBenar + prediksiSalah) * 100, "%.")
