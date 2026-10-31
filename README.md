# Prediksi Jumlah Kunjungan Wisatawan Mancanegara ke Indonesia per Bulan Menggunakan LSTM

Proyek TA-04 (Pembelajaran Mesin). Data resmi: Badan Pusat Statistik (BPS).
Alur: unduh data → EDA → MinMaxScaler → windowing → LSTM (TensorFlow/Keras) → RMSE & MAE → simpan model & scaler → aplikasi Flask → Render.

> Semua angka hasil dibuat oleh program dan disimpan di `outputs/metrics.json`. Tidak ada angka yang ditulis manual.

## Struktur folder

```
wisman-lstm/
├── app.py                  # aplikasi Flask
├── requirements.txt        # daftar library
├── Procfile, render.yaml   # konfigurasi deployment Render
├── .gitignore
├── data/
│   ├── raw/                # TARUH file unduhan BPS di sini (.csv / .xlsx)
│   └── processed/          # deret bulanan bersih (dibuat otomatis)
├── src/
│   ├── config.py           # hiperparameter & path
│   ├── data_loader.py      # pembaca file BPS
│   ├── inspect_data.py     # melihat isi file mentah
│   └── train.py            # EDA, preprocessing, pelatihan, evaluasi, simpan
├── models/                 # lstm_wisman.keras & scaler.joblib (dibuat otomatis)
├── outputs/                # metrics.json, forecast.json, figures/ (dibuat otomatis)
├── templates/              # HTML (Bootstrap 5)
├── static/css/style.css
└── tests/test_loader.py
```

## Panduan langkah demi langkah (Windows + VS Code)

### 1. Persiapan (sekali saja)
1. Pasang **Python 3.12 (64-bit)** dari python.org. Centang **"Add python.exe to PATH"** saat instalasi.
2. Pasang **Visual Studio Code**, lalu pasang ekstensi **Python** (Microsoft).
3. Pasang **Git** dari git-scm.com.
4. Ekstrak ZIP proyek ke folder tanpa spasi, misalnya `C:\proyek\wisman-lstm`.
5. Di VS Code: **File → Open Folder** → pilih `wisman-lstm`.
6. Buka terminal: **Terminal → New Terminal** (pilih PowerShell).

### 2. Membuat lingkungan virtual dan memasang library
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Jika muncul galat "running scripts is disabled", jalankan sekali:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, lalu ulangi aktivasi.
Jika VS Code menanyakan interpreter, pilih yang berada di folder `venv`
(**Ctrl+Shift+P → Python: Select Interpreter**).

### 3. Mengunduh dataset BPS
1. Buka tabel BPS: https://www.bps.go.id/id/statistics-table/2/MTE1MCMy/jumlah-kunjungan-wisatawan-mancanegara-per-bulan-ke-indonesia-menurut-pintu-masuk-2008-sekarang-kunjungan.html
2. Gunakan tombol **Unduh** (Excel/CSV) pada tabel.
3. Simpan file ke folder `data/raw/` (jangan diubah isinya).
4. Lihat isi file mentah:
   ```powershell
   python -m src.inspect_data
   ```
   Pastikan terlihat periode bulanan dan jumlah kunjungan. Jika struktur tidak dikenali,
   program akan menampilkan pesan jelas; rapikan file menjadi dua kolom (`Periode`, `Jumlah Kunjungan`).
5. Catat di laporan: tanggal unduh, rentang periode, dan perlakuan data (misal bulan hilang).

### 4. Melatih model
```powershell
python -m src.train
```
Keluaran otomatis: `data/processed/wisman_bulanan.csv`, 8 gambar di `outputs/figures/`,
`models/lstm_wisman.keras`, `models/scaler.joblib`, `outputs/metrics.json`, `outputs/forecast.json`,
`outputs/prediksi_data_uji.csv`. Pelatihan umumnya hanya beberapa menit di CPU.

### 5. Menjalankan aplikasi web
```powershell
python app.py
```
Buka http://127.0.0.1:5000 di browser. Hentikan dengan **Ctrl+C**.

### 6. Mengunggah ke GitHub
1. Buat repositori kosong di github.com (misal `prediksi-wisman-lstm`), tanpa README.
2. Di terminal VS Code:
```powershell
git init
git add .
git commit -m "Prediksi wisman dengan LSTM"
git branch -M main
git remote add origin https://github.com/USERNAME/prediksi-wisman-lstm.git
git push -u origin main
```
Folder `models/`, `outputs/`, dan `data/processed/` **harus ikut ter-commit** karena dipakai aplikasi di Render.

### 7. Deploy ke Render
1. Daftar di render.com, hubungkan akun GitHub.
2. **New → Web Service** → pilih repositori.
3. Isian: Runtime **Python 3**; Build Command `pip install -r requirements.txt`;
   Start Command `gunicorn app:app --workers 1 --threads 2 --timeout 180`
   (atau pilih **Blueprint** agar `render.yaml` terbaca otomatis).
4. Tekan **Create Web Service**, tunggu build selesai, lalu buka URL `.onrender.com`.
5. Jika layanan gratis kehabisan memori (TensorFlow cukup besar), tambahkan Environment Variable
   `USE_STATIC` = `1`. Aplikasi akan menampilkan peramalan hasil pelatihan yang tersimpan
   (`outputs/forecast.json`) tanpa memuat TensorFlow. Tuliskan mode yang dipakai di laporan.
6. Layanan gratis tertidur saat tidak dipakai; akses pertama bisa lambat.

## Mengubah hiperparameter
Edit `src/config.py` (mis. `LOOK_BACK`, `LSTM_UNITS`), jalankan ulang `python -m src.train`,
lalu commit dan push ulang agar Render memperbarui.

## Uji kecil
```powershell
python -m unittest discover tests
```
