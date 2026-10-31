"""Aplikasi web Flask: dashboard hasil dan peramalan kunjungan wisman berbasis LSTM.

Jalankan lokal : python app.py
Produksi Render: gunicorn app:app
Jika TensorFlow tidak dapat dimuat (mis. memori Render gratis terbatas), aplikasi
otomatis memakai outputs/forecast.json hasil pelatihan (mode statis).
"""
import json
import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src import config as C

app = Flask(__name__)

_state = {"model": None, "scaler": None, "mode": None, "error": None}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_model():
    """Muat model sekali (lazy). Kembalikan 'lstm' atau 'statis'."""
    if _state["mode"]:
        return _state["mode"]
    if os.environ.get("USE_STATIC") == "1":      # lewati TensorFlow (hemat memori di Render gratis)
        _state["mode"] = "statis"
        return "statis"
    try:
        import joblib
        from tensorflow.keras.models import load_model
        _state["model"] = load_model(C.MODEL_PATH)
        _state["scaler"] = joblib.load(C.SCALER_PATH)
        _state["mode"] = "lstm"
    except Exception as exc:  # noqa: BLE001
        _state["error"] = str(exc)
        _state["mode"] = "statis"
    return _state["mode"]


def _series():
    if not C.SERIES_CSV.exists():
        return None
    df = pd.read_csv(C.SERIES_CSV, parse_dates=["periode"])
    return df.set_index("periode")["jumlah_kunjungan"]


def forecast(horizon: int):
    s = _series()
    static = _read_json(C.FORECAST_PATH)
    if s is None or static is None:
        raise FileNotFoundError("Model belum dilatih. Jalankan `python -m src.train` terlebih dahulu.")
    if _load_model() == "statis":
        return static[:horizon], "statis"
    scaler, model = _state["scaler"], _state["model"]
    scaled = scaler.transform(s.values.reshape(-1, 1))[:, 0].tolist()
    out = []
    for _ in range(horizon):
        x = np.array(scaled[-C.LOOK_BACK:]).reshape(1, C.LOOK_BACK, 1)
        nxt = float(model.predict(x, verbose=0)[0, 0])
        scaled.append(nxt)
        out.append(nxt)
    vals = np.maximum(scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel(), 0)
    idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return [{"periode": d.strftime("%Y-%m"), "prediksi": float(v)} for d, v in zip(idx, vals)], "lstm"


@app.route("/")
def index():
    metrics = _read_json(C.METRICS_PATH)
    s = _series()
    history = None if s is None else {"periode": s.index.strftime("%Y-%m").tolist(), "nilai": s.round(0).tolist()}
    return render_template("index.html", metrics=metrics, history=history,
                           ready=metrics is not None and s is not None, horizon_max=C.FORECAST_HORIZON_MAX)


@app.route("/api/forecast")
def api_forecast():
    try:
        h = int(request.args.get("horizon", 6))
    except ValueError:
        return jsonify(error="horizon harus berupa bilangan bulat"), 400
    if not 1 <= h <= C.FORECAST_HORIZON_MAX:
        return jsonify(error=f"horizon harus 1 sampai {C.FORECAST_HORIZON_MAX}"), 400
    try:
        data, mode = forecast(h)
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503
    return jsonify(horizon=h, mode=mode, forecast=data)


@app.route("/about")
def about():
    return render_template("about.html", metrics=_read_json(C.METRICS_PATH))


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
