
"""Membaca dataset BPS wisatawan mancanegara menjadi deret waktu bulanan."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DIR

BULAN = {
    "januari": 1, "jan": 1,
    "februari": 2, "feb": 2,
    "maret": 3, "mar": 3,
    "april": 4, "apr": 4,
    "mei": 5, "may": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "agustus": 8, "agu": 8, "agt": 8,
    "september": 9, "sep": 9,
    "oktober": 10, "okt": 10,
    "november": 11, "nov": 11,
    "desember": 12, "des": 12,
}


def to_number(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.number)):
        return float(x)

    s = str(x).strip().replace(" ", "")

    if s in ("", "-", "–", "...", "nan", "None"):
        return np.nan

    if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", s):
        return float(re.sub(r"[.,]", "", s))

    try:
        return float(s.replace(",", "."))
    except ValueError:
        return np.nan


def _read_raw(path):
    path = Path(path)

    if path.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, header=None, dtype=object)

    return pd.read_csv(
        path,
        header=None,
        dtype=object,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )


def _from_bps_wide(raw):
    """
    Membaca format BPS:
    - Baris berisi nama bulan.
    - Kolom pertama berisi nama pintu masuk.
    - Baris Total digunakan sebagai data bulanan.
    """

    for i in range(min(len(raw), 20)):

        header = (
            raw.iloc[i]
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )

        month_cols = {}

        for j, value in enumerate(header):
            if value in BULAN:
                month_cols[j] = BULAN[value]

        # Minimal satu bulan dikenali
        if len(month_cols) < 1:
            continue

        body = raw.iloc[i + 1:].copy()

        labels = (
            body.iloc[:, 0]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Cari baris Total, bukan subtotal pintu masuk
        total_mask = labels.str.fullmatch(
            r"total",
            na=False
        )

        if not total_mask.any():
            continue

        total_row = body.loc[total_mask].iloc[-1]

        year = None

        # Cari tahun pada baris sebelum header bulan
        for k in range(i - 1, -1, -1):
            values = raw.iloc[k].astype(str).tolist()

            for value in values:
                match = re.search(r"\b(19|20)\d{2}\b", value)

                if match:
                    year = int(match.group())
                    break

            if year is not None:
                break

        if year is None:
            raise ValueError("Tahun dataset tidak ditemukan.")

        dates = []
        values = []

        for col, month in month_cols.items():

            value = to_number(total_row.iloc[col])

            # Bulan yang belum memiliki data dilewati
            if pd.isna(value):
                continue

            dates.append(pd.Timestamp(year, month, 1))
            values.append(value)

        if len(values) == 0:
            continue

        series = pd.Series(
            values,
            index=pd.DatetimeIndex(dates),
            name="jumlah_kunjungan"
        )

        return series, "baris Total dataset BPS"

    return None



def load_series(path):
    import pandas as pd

    df = pd.read_csv(path)

    # Pastikan kolom date dan value tersedia
    if not {"date", "value"}.issubset(df.columns):
        raise ValueError(
            "Dataset harus memiliki kolom date dan value."
        )

    # Ambil tanggal dan jumlah kunjungan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "value"])

    # Urutkan berdasarkan tanggal
    df = df.sort_values("date")

    # Jadikan tanggal sebagai index
    series = df.set_index("date")["value"]

    # Pastikan data bulanan
    series = series.resample("MS").mean()

    # Isi bulan kosong jika ada
    series = series.interpolate(method="linear")

    return series



def find_raw_file():
    files = [
        p for p in RAW_DIR.glob("*")
        if p.suffix.lower() in (".csv", ".xlsx", ".xls", ".xlsm")
    ]

    if not files:
        raise FileNotFoundError(
            "Letakkan dataset BPS di folder data/raw/"
        )

    return max(files, key=lambda p: p.stat().st_mtime)
