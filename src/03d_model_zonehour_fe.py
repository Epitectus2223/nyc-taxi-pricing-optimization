import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

DB_NAME = "taxi_pricing_v2.db"
TABLE = "hourly_zone_panel"
TRAIN_END = "2023-03-01"
MIN_TRIPS_TRAIN = 10

OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_NAME)
df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
conn.close()

df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
df["PULocationID"] = df["PULocationID"].astype(int)

# Features core
df["log_price"] = np.log(df["avg_fare_per_mile"])
df["avg_speed_mph"] = df["avg_distance"] / (df["avg_duration_min"] / 60.0)
df = df[(df["avg_speed_mph"] > 1) & (df["avg_speed_mph"] < 60)].copy()

df["log_distance"] = np.log(df["avg_distance"].clip(lower=1e-3))
df["log_duration"] = np.log(df["avg_duration_min"].clip(lower=1e-3))
df["log_speed"] = np.log(df["avg_speed_mph"].clip(lower=1e-3))

# Split
train = df[(df["pickup_hour"] < TRAIN_END) & (df["trips"] >= MIN_TRIPS_TRAIN)].copy()
test  = df[(df["pickup_hour"] >= TRAIN_END)].copy()

# --- Center price within zone-hour (train means, evita leakage) ---
zh_mean = train.groupby(["PULocationID", "hour"])["log_price"].mean()
train["log_price_zh"] = train["log_price"] - train.set_index(["PULocationID","hour"]).index.map(zh_mean)
test["log_price_zh"]  = test["log_price"]  - test.set_index(["PULocationID","hour"]).index.map(zh_mean)
test["log_price_zh"] = test["log_price_zh"].fillna(0.0)

# Categorías consistentes (evita patsy error)
train_zones = sorted(train["PULocationID"].unique())
train["PULocationID"] = pd.Categorical(train["PULocationID"], categories=train_zones)
test["PULocationID"]  = pd.Categorical(test["PULocationID"],  categories=train_zones)
before = len(test)
test = test.dropna(subset=["PULocationID"]).copy()
print(f"Filas test removidas por zonas no vistas en train: {before-len(test):,}")

# Modelo con FE zona×hora
formula = """
trips ~ log_price_zh
      + log_distance + log_duration + log_speed
      + rush_hour + is_weekend
      + C(day_of_week)
      + C(PULocationID):C(hour)
"""

model = smf.glm(
    formula=formula,
    data=train,
    family=sm.families.NegativeBinomial(alpha=1.0)
).fit()

print(model.summary())

pred = model.predict(test)
y_test = test["trips"].astype(float)

mae = np.mean(np.abs(y_test - pred))
rmse = np.sqrt(np.mean((y_test - pred)**2))

print("\nPred describe:\n", pd.Series(pred).describe())
print("\nTest MAE:", mae, " RMSE:", rmse)
print("coef_log_price_zh:", model.params.get("log_price_zh", np.nan))

pd.DataFrame([{
    "model":"negbin_zonehour_fe",
    "MAE":mae,
    "RMSE":rmse,
    "coef_log_price_zh": model.params.get("log_price_zh", np.nan)
}]).to_csv(OUT_DIR / "glm_zonehour_fe_metrics.csv", index=False)
