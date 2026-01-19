import pandas as pd
import numpy as np
import sqlite3

DB_NAME = "taxi_pricing_v2.db"
SOURCE_TABLE = "taxi_trips_cleaned_v2"
TARGET_TABLE = "hourly_zone_panel"


LOW_TRIPS_THRESHOLD = 10

conn = sqlite3.connect(DB_NAME)

cols_in_db = pd.read_sql(f"PRAGMA table_info({SOURCE_TABLE});", conn)
cols_in_db = cols_in_db["name"].tolist()

needed = [
    "pickup_hour", "PULocationID",
    "fare_per_mile", "trip_distance", "trip_duration_min",
    "hour", "day_of_week", "is_weekend", "rush_hour",
    "payment_type"
]
available = [c for c in needed if c in cols_in_db]

query = f"SELECT {', '.join(available)} FROM {SOURCE_TABLE}"
df = pd.read_sql(query, conn)
conn.close()

print(f"Filas cargadas desde {SOURCE_TABLE}: {len(df):,}")
print("Columnas disponibles:", available)


df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])

group_keys = ["pickup_hour", "PULocationID"]

agg_dict = {
    "fare_per_mile": ["mean", "median"],
    "trip_distance": "mean",
    "trip_duration_min": "mean",
    "hour": "first",
    "day_of_week": "first",
    "is_weekend": "first",
    "rush_hour": "first"
}

panel = (
    df.groupby(group_keys)
      .agg(agg_dict)
)

panel.columns = [
    "avg_fare_per_mile" if c == ("fare_per_mile", "mean") else
    "median_fare_per_mile" if c == ("fare_per_mile", "median") else
    "avg_distance" if c == ("trip_distance", "mean") else
    "avg_duration_min" if c == ("trip_duration_min", "mean") else
    c[0] if isinstance(c, tuple) else c
    for c in panel.columns
]
panel = panel.reset_index()

panel["trips"] = df.groupby(group_keys).size().values

if "payment_type" in df.columns:
    pt = df.copy()
    pt["is_cash"] = (pt["payment_type"] == 2).astype(int)
    pt["is_card"] = (pt["payment_type"] == 1).astype(int)

    shares = (
        pt.groupby(group_keys)
          .agg(share_cash=("is_cash", "mean"),
               share_card=("is_card", "mean"))
          .reset_index()
    )

    panel = panel.merge(shares, on=group_keys, how="left")
else:
    panel["share_cash"] = np.nan
    panel["share_card"] = np.nan

panel["low_trips_flag"] = (panel["trips"] < LOW_TRIPS_THRESHOLD).astype(int)

print("\n=== Sanity checks ===")
print("Panel filas (hora×zona):", len(panel))
print("Rango fechas:", panel["pickup_hour"].min(), "→", panel["pickup_hour"].max())
print("\nResumen trips:")
print(panel["trips"].describe())

print("\nTop 10 horas-zona por trips:")
print(panel.sort_values("trips", ascending=False).head(10)[
    ["pickup_hour", "PULocationID", "trips", "avg_fare_per_mile"]
])

print("\nBottom 10 horas-zona por trips:")
print(panel.sort_values("trips", ascending=True).head(10)[
    ["pickup_hour", "PULocationID", "trips", "avg_fare_per_mile", "low_trips_flag"]
])

conn = sqlite3.connect(DB_NAME)
panel.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
conn.close()

panel.to_csv("hourly_zone_panel.csv", index=False)
print(f"\n Guardado en SQLite: {DB_NAME}::{TARGET_TABLE}")
print("Exportado para Power BI: hourly_zone_panel.csv")


