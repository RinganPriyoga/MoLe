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

# --- 2. ALGORITMA REGRESI LOGISTIK MANUAL ---
class RegresiLogistikManual:
    def __init__(self, learning_rate=0.01, iterasi=1000):
        self.lr = learning_rate
        self.iterasi = iterasi
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iterasi):
            model_linear = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(model_linear)

            # Perhitungan turunan (gradien) dari log loss
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameter
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        model_linear = np.dot(X, self.w) + self.b
        probabilitas = self.sigmoid(model_linear)
        return np.array([1 if p > 0.5 else 0 for p in probabilitas])

# --- 3. EKSEKUSI PROGRAM ---
if __name__ == "__main__":
    # Nanti sesuaikan nama file dan nama kolom ini dengan file aslimu
    NAMA_FILE = 'data_logistik.csv' 
    KOLOM_X = 'X' 
    KOLOM_Y = 'y_kelas' # Kolom target harus berisi angka 0 dan 1
    
    X, y = muat_data_csv(NAMA_FILE, KOLOM_X, KOLOM_Y)
    
    if X is not None:
        model = RegresiLogistikManual(learning_rate=0.1, iterasi=2000)
        model.fit(X, y)
        
        # Buat titik-titik kurva yang berurutan untuk plotting
        X_plot = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
        prob_plot = model.sigmoid(np.dot(X_plot, model.w) + model.b)

        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='blue', label='Data Aktual (Kelas 0 & 1)')
        plt.plot(X_plot, prob_plot, color='red', linewidth=2, label='Kurva Sigmoid Manual')
        plt.axhline(0.5, color='green', linestyle='--', label='Threshold 0.5')
        plt.title('Regresi Logistik (Algoritma Manual)')
        plt.xlabel(KOLOM_X)
        plt.ylabel('Probabilitas')
        plt.legend()
        plt.grid(True)
        plt.show()