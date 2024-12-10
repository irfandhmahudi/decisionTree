import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Membaca dataset
data = pd.read_csv('gender.csv', delimiter=',')

# Menghilangkan spasi di nama kolom
data.columns = data.columns.str.strip()

# Label encoding untuk kolom kategori
label_encoders = {}
for column in ["Gender", "Occupation", "Education Level", "Marital Status", "Favorite Color"]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Menyimpan encoder untuk referensi jika diperlukan

# Menentukan fitur dan label
input_features = data.drop(columns=["Gender"]).to_numpy()  # Semua kolom kecuali "Gender"
labels = data["Gender"].to_numpy()  # "Gender" sebagai label

# Memeriksa jumlah label unik
print("Unique labels in Gender:", np.unique(labels))

# Membagi dataset (menggunakan data yang sama untuk demo)
inputTraining = input_features
labelTraining = labels
inputTesting = input_features
labelTesting = labels

# Melatih Decision Tree Classifier
model = tree.DecisionTreeClassifier(random_state=0)
model.fit(inputTraining, labelTraining)

# Memprediksi data uji
hasilPrediksi = model.predict(inputTesting)

# Menghitung akurasi
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
akurasi = prediksiBenar / (prediksiBenar + prediksiSalah) * 100

# Menampilkan hasil evaluasi
print("Hasil prediksi: ", hasilPrediksi)
print("Label sebenarnya: ", labelTesting)
print("Prediksi benar: ", prediksiBenar, "data")
print("Prediksi salah: ", prediksiSalah, "data")
print("Akurasi: {:.2f}%".format(akurasi))

