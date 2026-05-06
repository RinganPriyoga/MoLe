import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cek_statistik_dasar(df, nama_kolom):
    """Fungsi untuk EDA (Exploratory Data Analysis)"""
    print(f"--- Statistik Kolom '{nama_kolom}' ---")
    print(f"Mean   : {np.mean(df[nama_kolom]):.4f}")
    print(f"Median : {np.median(df[nama_kolom]):.4f}")
    print(f"Modus  : {df[nama_kolom].mode().values[0]}\n")

class RegresiLogistikManual:
    def __init__(self, learning_rate=0.01, iterasi=1000):
        self.lr = learning_rate
        self.iterasi = iterasi
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterasi):
            model_linear = np.dot(X, self.weights) + self.bias
            y_prediksi = self.sigmoid(model_linear)

            # Menghitung gradien berdasarkan Cost Function (Log Loss)
            dw = (1 / n_samples) * np.dot(X.T, (y_prediksi - y))
            db = (1 / n_samples) * np.sum(y_prediksi - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        model_linear = np.dot(X, self.weights) + self.bias
        y_prediksi_probabilitas = self.sigmoid(model_linear)
        # Mengubah probabilitas menjadi kelas 0 atau 1 berdasarkan threshold 0.5
        return np.array([1 if p > 0.5 else 0 for p in y_prediksi_probabilitas])

if __name__ == "__main__":
    # 1. BACA FILE CSV
    # GANTI 'data_logistik.csv' dengan nama file kamu yang sebenarnya
    nama_file = 'data_logistik.csv' 
    df = pd.read_csv(nama_file)
    
    # GANTI 'NamaKolomX' dan 'NamaKolomTargetBiner' sesuai header di CSV kamu
    # Pastikan kolom target nilainya sudah berupa angka 0 dan 1
    kolom_fitur = 'NamaKolomX' 
    kolom_target = 'NamaKolomTargetBiner' 

    # Menampilkan statistik dasar
    cek_statistik_dasar(df, kolom_fitur)

    # 2. PERSIAPAN DATA
    X = df[[kolom_fitur]].values 
    y = df[kolom_target].values

    # 3. TRAINING MODEL
    model = RegresiLogistikManual(learning_rate=0.1, iterasi=2000)
    model.fit(X, y)
    
    # 4. MEMBUAT GARIS SIGMOID UNTUK PLOTTING
    # Kita membuat titik-titik X berurutan agar kurva terlihat mulus (smooth)
    X_plot = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    probabilitas_plot = model.sigmoid(np.dot(X_plot, model.weights) + model.bias)

    # 5. VISUALISASI
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', label='Data Aktual (CSV) Kelas 0 & 1')
    plt.plot(X_plot, probabilitas_plot, color='red', linewidth=2, label='Kurva Probabilitas Sigmoid')
    
    # Garis batas (Threshold)
    plt.axhline(0.5, color='green', linestyle='--', label='Threshold 0.5')
    
    plt.title('Regresi Logistik Manual dari CSV')
    plt.xlabel(kolom_fitur)
    plt.ylabel('Probabilitas / Kelas Target')
    plt.legend()
    plt.grid(True)
    plt.show()