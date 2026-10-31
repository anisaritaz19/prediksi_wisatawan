"""Melihat isi mentah file BPS: python -m src.inspect_data [path_file]"""
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import _read_raw, find_raw_file

path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_raw_file()
raw = _read_raw(path)
pd.set_option("display.width", 200, "display.max_columns", 14, "display.max_colwidth", 25)
print("File:", path, "| ukuran:", raw.shape)
print(raw.head(12))
print("...")
print(raw.tail(5))
