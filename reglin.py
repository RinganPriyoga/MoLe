import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. FUNGSI BACA CSV ---
def muat_data_csv(nama_file, nama_kolom_x, nama_kolom_y):
    print(f"Membaca data dari: {nama_file}...")
    try:
        df = pd.read_csv(nama_file)
        X = df[[nama_kolom_x]].values 
        y = df[nama_kolom_y].values
        print(f"Berhasil! Ditemukan {len(df)} baris data.\n")
        return X, y
    except FileNotFoundError:
        print(f"Error: File '{nama_file}' tidak ditemukan.")
        return None, None

# --- 2. ALGORITMA REGRESI LINEAR MANUAL ---
class RegresiLinearManual:
    def __init__(self, learning_rate=0.01, iterasi=1000):
        self.lr = learning_rate
        self.iterasi = iterasi
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iterasi):
            y_pred = np.dot(X, self.w) + self.b
            
            # Perhitungan turunan (gradien)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameter
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

# --- 3. EKSEKUSI PROGRAM ---
if __name__ == "__main__":
    # Nanti sesuaikan nama file dan nama kolom ini dengan file aslimu
    NAMA_FILE = 'data_linear.csv' 
    KOLOM_X = 'X' 
    KOLOM_Y = 'y'
    
    X, y = muat_data_csv(NAMA_FILE, KOLOM_X, KOLOM_Y)
    
    if X is not None:
        model = RegresiLinearManual(learning_rate=0.01, iterasi=1000)
        model.fit(X, y)
        prediksi = model.predict(X)

        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='blue', label='Data Aktual')
        plt.plot(X, prediksi, color='red', linewidth=2, label='Garis Regresi Manual')
        plt.title('Regresi Linear (Algoritma Manual)')
        plt.xlabel(KOLOM_X)
        plt.ylabel(KOLOM_Y)
        plt.legend()
        plt.grid(True)
        plt.show()