# 01_build_clean_db_v2.py
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

DATA_DIR = Path(".")  # carpeta donde están los parquet
FILES = [
    "yellow_tripdata_2023-01.parquet",
    "yellow_tripdata_2023-02.parquet",
    "yellow_tripdata_2023-03.parquet",
]

DB_NAME = "taxi_pricing_v2.db"
TABLE_NAME = "taxi_trips_cleaned_v2"

# Columnas a leer (agrega/quita según disponibilidad en tu parquet)
COLS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID",
    "trip_distance", "fare_amount", "total_amount",
    "payment_type", "passenger_count",
    "RatecodeID", "tip_amount", "tolls_amount",
    "congestion_surcharge", "airport_fee"
]

def read_parquet_safe(file_path: Path, cols: list[str]) -> pd.DataFrame:
    # Lee solo las columnas disponibles (por si algunas no existen)
    tmp = pd.read_parquet(file_path)
    available = [c for c in cols if c in tmp.columns]
    return pd.read_parquet(file_path, columns=available)

df_list = []
for f in FILES:
    fp = DATA_DIR / f
    print(f"Leyendo: {fp}")
    df_list.append(read_parquet_safe(fp, COLS))

df = pd.concat(df_list, ignore_index=True)
print("Filas originales:", len(df))

# --- Timestamps ---
df["pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
df["dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

# --- Filtro por rango temporal ---
START = "2023-01-01"
END = "2023-04-01"  # exclusivo
df = df[(df["pickup_datetime"] >= START) & (df["pickup_datetime"] < END)].copy()

# --- Duración ---
df["trip_duration_min"] = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60

# --- Features temporales ---
df["pickup_hour"] = df["pickup_datetime"].dt.floor("h")
df["hour"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

# --- Reglas base ---
base_mask = (
    (df["fare_amount"] > 0) &
    (df["total_amount"] > 0) &
    (df["trip_distance"] > 0) &
    (df["trip_duration_min"] >= 1) &
    (df["trip_duration_min"] <= 180) &
    (df["PULocationID"].notna())
)

df = df.loc[base_mask].copy()
print("Filas tras reglas base:", len(df))

# --- Precio por milla ---
# Evita explosiones por distancias diminutas
MIN_DIST = 0.5
df = df[df["trip_distance"] >= MIN_DIST].copy()

df["fare_per_mile"] = df["fare_amount"] / df["trip_distance"]

# --- Recorte por percentiles globales (v1). Luego podemos hacerlo por zona. ---
p01, p99 = df["fare_per_mile"].quantile([0.01, 0.99])
df = df[(df["fare_per_mile"] >= p01) & (df["fare_per_mile"] <= p99)].copy()

print("Filas tras recorte fare_per_mile (P1-P99):", len(df))
print("fare_per_mile P1-P99:", float(p01), float(p99))

# --- Guardar a SQLite ---
conn = sqlite3.connect(DB_NAME)
df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
conn.close()
print(f"Guardado en {DB_NAME}::{TABLE_NAME}")

# --- Reporte de calidad (para tu README) ---
quality = pd.DataFrame({
    "metric": ["rows_original", "rows_after_base_rules", "rows_after_min_dist", "rows_after_fpmile_clip"],
    "value": [sum(len(x) for x in df_list), None, None, len(df)]
})
quality.loc[1, "value"] = "ver consola"
quality.loc[2, "value"] = f"MIN_DIST={MIN_DIST}"
quality.to_csv("quality_report_step1.csv", index=False)
print("Reporte de calidad: quality_report_step1.csv")

