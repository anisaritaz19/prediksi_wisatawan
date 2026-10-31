"""Pipeline lengkap: baca data -> EDA -> preprocessing -> LSTM -> evaluasi -> simpan.

Jalankan dari folder proyek:
    python -m src.train                 # memakai file terbaru di data/raw/
    python -m src.train path/ke/file.csv
Semua angka hasil (RMSE, MAE, dst.) ditulis ke outputs/metrics.json; tidak ada angka
yang ditulis manual.
"""
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from src import config as C
from src.data_loader import find_raw_file, load_series

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})
BLACK_WHITE = {"line": "black", "line2": "dimgray"}   # jurnal dicetak hitam-putih


def savefig(name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / name, dpi=200)
    plt.close()


# ----------------------------------------------------------------- EDA
def run_eda(s: pd.Series) -> dict:
    stats = s.describe().to_dict()
    stats["skewness"] = float(s.skew())
    stats["n_bulan"] = int(len(s))
    stats["bulan_min"] = s.idxmin().strftime("%Y-%m")
    stats["bulan_maks"] = s.idxmax().strftime("%Y-%m")

    plt.figure(figsize=(7, 3))
    plt.plot(s.index, s.values, color="black", lw=1)
    plt.xlabel("Periode"); plt.ylabel("Jumlah kunjungan")
    savefig("01_deret_waktu.png")

    roll = s.rolling(12).mean()
    plt.figure(figsize=(7, 3))
    plt.plot(s.index, s.values, color="darkgray", lw=0.8, label="Data bulanan")
    plt.plot(roll.index, roll.values, color="black", lw=1.5, label="Rata-rata bergerak 12 bulan")
    plt.legend(); plt.xlabel("Periode"); plt.ylabel("Jumlah kunjungan")
    savefig("02_rata_rata_bergerak.png")

    month_mean = s.groupby(s.index.month).mean()
    plt.figure(figsize=(7, 3))
    plt.bar(month_mean.index, month_mean.values, color="gray", edgecolor="black")
    plt.xticks(range(1, 13)); plt.xlabel("Bulan"); plt.ylabel("Rata-rata kunjungan")
    savefig("03_pola_bulanan.png")

    yearly = s.groupby(s.index.year).sum()
    plt.figure(figsize=(7, 3))
    plt.bar(yearly.index.astype(str), yearly.values, color="gray", edgecolor="black")
    plt.xticks(rotation=60); plt.xlabel("Tahun"); plt.ylabel("Total kunjungan per tahun")
    savefig("04_total_tahunan.png")

    plt.figure(figsize=(4, 3))
    plt.hist(s.values, bins=20, color="gray", edgecolor="black")
    plt.xlabel("Jumlah kunjungan"); plt.ylabel("Frekuensi")
    savefig("05_histogram.png")
    return stats


# ----------------------------------------------------------------- windowing
def make_windows(arr: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(look_back, len(arr)):
        X.append(arr[i - look_back:i, 0])
        y.append(arr[i, 0])
    return np.array(X)[..., None], np.array(y)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def main(path=None):
    random.seed(C.SEED); np.random.seed(C.SEED)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(C.SEED)
    from tensorflow.keras import callbacks, layers, models, optimizers

    path = Path(path) if path else find_raw_file()
    print(f"[data] file: {path}")
    s = load_series(path)
    for d in (C.PROCESSED_DIR, C.MODEL_DIR, C.FIG_DIR, C.OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    s.rename("jumlah_kunjungan").to_csv(C.SERIES_CSV, index_label="periode")

    eda = run_eda(s)

    # ---- pembagian kronologis; scaler hanya di-fit pada data latih (mencegah kebocoran data)
    values = s.values.reshape(-1, 1)
    n = len(values)
    n_test = int(round(n * C.TEST_RATIO))
    n_train_total = n - n_test
    n_val = int(round(n_train_total * C.VAL_RATIO))
    n_train = n_train_total - n_val
    if n_train <= C.LOOK_BACK + 5:
        raise ValueError("Data terlalu pendek untuk LOOK_BACK yang dipilih.")

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(values[:n_train])
    scaled = scaler.transform(values)

    # jendela prediksi memakai LOOK_BACK bulan sebelumnya yang boleh berasal dari segmen sebelumnya
    X_all, y_all = make_windows(scaled, C.LOOK_BACK)
    target_idx = np.arange(C.LOOK_BACK, n)
    tr = target_idx < n_train
    va = (target_idx >= n_train) & (target_idx < n_train_total)
    te = target_idx >= n_train_total
    X_tr, y_tr, X_va, y_va, X_te, y_te = X_all[tr], y_all[tr], X_all[va], y_all[va], X_all[te], y_all[te]
    print(f"[split] latih={len(X_tr)} validasi={len(X_va)} uji={len(X_te)} jendela | look_back={C.LOOK_BACK}")

    # ---- model
    model = models.Sequential([
        layers.Input(shape=(C.LOOK_BACK, 1)),
        layers.LSTM(C.LSTM_UNITS),
        layers.Dropout(C.DROPOUT),
        layers.Dense(1),
    ])
    model.compile(optimizer=optimizers.Adam(C.LEARNING_RATE), loss="mse")
    model.summary()
    cb = [callbacks.EarlyStopping(monitor="val_loss", patience=C.PATIENCE, restore_best_weights=True)]
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=C.EPOCHS,
                     batch_size=C.BATCH_SIZE, shuffle=False, callbacks=cb, verbose=2)

    plt.figure(figsize=(5, 3))
    plt.plot(hist.history["loss"], color="black", label="Loss latih")
    plt.plot(hist.history["val_loss"], color="gray", ls="--", label="Loss validasi")
    plt.xlabel("Epoch"); plt.ylabel("MSE (skala 0-1)"); plt.legend()
    savefig("06_kurva_loss.png")

    # ---- evaluasi pada skala asli
    def inv(a):
        return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()

    res = {}
    for name, X, y in (("latih", X_tr, y_tr), ("validasi", X_va, y_va), ("uji", X_te, y_te)):
        pred, true = inv(model.predict(X, verbose=0)), inv(y)
        res[name] = {"RMSE": rmse(true, pred), "MAE": float(mean_absolute_error(true, pred)), "n": int(len(true))}
    print("[evaluasi]", json.dumps(res, indent=2))

    # pembanding sederhana (naive: bulan ini = bulan lalu) pada data uji agar hasil LSTM bisa dinilai wajar
    naive_pred = inv(X_te[:, -1, 0]); true_te = inv(y_te)
    res["naive_uji"] = {"RMSE": rmse(true_te, naive_pred), "MAE": float(mean_absolute_error(true_te, naive_pred))}

    # ---- visualisasi hasil
    idx = s.index
    p_tr, p_va, p_te = (inv(model.predict(X, verbose=0)) for X in (X_tr, X_va, X_te))
    plt.figure(figsize=(7, 3))
    plt.plot(idx, s.values, color="black", lw=1, label="Aktual")
    plt.plot(idx[target_idx[tr]], p_tr, color="dimgray", lw=1, ls=":", label="Prediksi (latih)")
    plt.plot(idx[target_idx[va]], p_va, color="gray", lw=1, ls="-.", label="Prediksi (validasi)")
    plt.plot(idx[target_idx[te]], p_te, color="black", lw=1.4, ls="--", label="Prediksi (uji)")
    plt.legend(fontsize=7); plt.xlabel("Periode"); plt.ylabel("Jumlah kunjungan")
    savefig("07_aktual_vs_prediksi.png")

    plt.figure(figsize=(7, 3))
    plt.plot(idx[target_idx[te]], true_te, color="black", lw=1.2, marker="o", ms=3, label="Aktual")
    plt.plot(idx[target_idx[te]], p_te, color="dimgray", lw=1.2, ls="--", marker="s", ms=3, label="Prediksi LSTM")
    plt.legend(); plt.xlabel("Periode"); plt.ylabel("Jumlah kunjungan")
    savefig("08_data_uji.png")

    # ---- peramalan rekursif ke depan
    last = scaled[-C.LOOK_BACK:, 0].tolist()
    fut = []
    for _ in range(C.FORECAST_HORIZON_MAX):
        nxt = float(model.predict(np.array(last[-C.LOOK_BACK:]).reshape(1, C.LOOK_BACK, 1), verbose=0)[0, 0])
        fut.append(nxt); last.append(nxt)
    fut_vals = np.maximum(inv(fut), 0)
    fut_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=C.FORECAST_HORIZON_MAX, freq="MS")
    forecast = [{"periode": d.strftime("%Y-%m"), "prediksi": float(v)} for d, v in zip(fut_idx, fut_vals)]

    # ---- simpan
    model.save(C.MODEL_PATH)
    joblib.dump(scaler, C.SCALER_PATH)
    metrics = {
        "sumber_file": path.name,
        "periode_awal": s.index[0].strftime("%Y-%m"), "periode_akhir": s.index[-1].strftime("%Y-%m"),
        "eda": eda,
        "split": {"n_latih_jendela": int(len(X_tr)), "n_validasi_jendela": int(len(X_va)), "n_uji_jendela": int(len(X_te))},
        "hyperparameter": {"look_back": C.LOOK_BACK, "lstm_units": C.LSTM_UNITS, "dropout": C.DROPOUT,
                           "epochs_maks": C.EPOCHS, "epochs_dijalankan": len(hist.history["loss"]),
                           "batch_size": C.BATCH_SIZE, "learning_rate": C.LEARNING_RATE,
                           "optimizer": "Adam", "loss": "MSE", "patience": C.PATIENCE, "seed": C.SEED},
        "hasil": res,
        "versi": {"tensorflow": tf.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "terakhir_data": float(s.iloc[-1]),
    }
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    C.FORECAST_PATH.write_text(json.dumps(forecast, indent=2), encoding="utf-8")
    pd.DataFrame({"periode": idx[target_idx[te]].strftime("%Y-%m"), "aktual": true_te, "prediksi": p_te}
                 ).to_csv(C.OUT_DIR / "prediksi_data_uji.csv", index=False)
    print(f"[selesai] model: {C.MODEL_PATH.name} | scaler: {C.SCALER_PATH.name} | metrik: {C.METRICS_PATH.name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
