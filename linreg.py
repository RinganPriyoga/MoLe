import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cek_statistik_dasar(df, nama_kolom):
    """Fungsi untuk EDA (Exploratory Data Analysis)"""
    print(f"--- Statistik Kolom '{nama_kolom}' ---")
    print(f"Mean   : {np.mean(df[nama_kolom]):.4f}")
    print(f"Median : {np.median(df[nama_kolom]):.4f}")
    print(f"Modus  : {df[nama_kolom].mode().values[0]}\n")

class RegresiLinearManual:
    def __init__(self, learning_rate=0.01, iterasi=1000):
        self.lr = learning_rate
        self.iterasi = iterasi
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterasi):
            y_prediksi = np.dot(X, self.weights) + self.bias
            # Menghitung gradien berdasarkan Cost Function (MSE)
            dw = (1 / n_samples) * np.dot(X.T, (y_prediksi - y))
            db = (1 / n_samples) * np.sum(y_prediksi - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # 1. BACA FILE CSV
    # GANTI 'data_linear.csv' dengan nama file kamu yang sebenarnya
    nama_file = 'data_linear.csv' 
    df = pd.read_csv(nama_file)
    
    # GANTI 'NamaKolomX' dan 'NamaKolomY' sesuai header di CSV kamu
    kolom_fitur = 'NamaKolomX' 
    kolom_target = 'NamaKolomY'

    # Menampilkan statistik dasar
    cek_statistik_dasar(df, kolom_fitur)
    cek_statistik_dasar(df, kolom_target)

    # 2. PERSIAPAN DATA (X harus berupa matriks 2D, y berupa vektor 1D)
    X = df[[kolom_fitur]].values 
    y = df[kolom_target].values

    # 3. TRAINING MODEL
    model = RegresiLinearManual(learning_rate=0.01, iterasi=1000)
    model.fit(X, y)
    
    # 4. PREDIKSI
    prediksi = model.predict(X)

    # 5. VISUALISASI
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', label='Data Aktual (CSV)')
    plt.plot(X, prediksi, color='red', linewidth=2, label='Garis Regresi')
    plt.title('Regresi Linear Manual dari CSV')
    plt.xlabel(kolom_fitur)
    plt.ylabel(kolom_target)
    plt.legend()
    plt.grid(True)
    plt.show()