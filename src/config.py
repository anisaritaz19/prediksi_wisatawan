"""Konfigurasi terpusat proyek. Ubah nilai di sini, bukan di banyak file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DIR = ROOT / "outputs"

SERIES_CSV = PROCESSED_DIR / "wisman_bulanan.csv"
MODEL_PATH = MODEL_DIR / "lstm_wisman.keras"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METRICS_PATH = OUT_DIR / "metrics.json"
FORECAST_PATH = OUT_DIR / "forecast.json"

# Hiperparameter (semuanya dicatat ke metrics.json agar laporan mengutipnya apa adanya)
LOOK_BACK = 12          # jumlah bulan sebelumnya yang dipakai untuk memprediksi 1 bulan
TEST_RATIO = 0.2        # porsi data uji (kronologis, bagian paling akhir)
VAL_RATIO = 0.1         # porsi validasi dari data latih (kronologis)
LSTM_UNITS = 64
DROPOUT = 0.2
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
PATIENCE = 20           # EarlyStopping
SEED = 42
FORECAST_HORIZON_MAX = 12
