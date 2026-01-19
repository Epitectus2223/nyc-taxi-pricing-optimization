import pandas as pd
import numpy as np
import sqlite3

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor

DB_NAME = "taxi_pricing_v2.db"
TABLE = "hourly_zone_panel"

TRAIN_END = "2023-03-01"
MIN_TRIPS_TRAIN = 10

MIN_TRIPS_FOR_RECOMMEND = 20

FACTORS = np.linspace(0.8, 1.2, 21)

POISSON_ALPHA = 1e-3
POISSON_MAX_ITER = 8000
POISSON_TOL = 1e-6

TOP_CORE_ZONES = 30      
TOP_MID_ZONES = 100      

ELASTICITY_BY_SEGMENT = {
    ("core", 0): [-0.3, -0.6, -0.9],       
    ("core", 1): [-1.2, -1.4, -1.6],        

    
    ("mid", 0): [-0.5, -0.8, -1.1],         
    ("mid", 1): [-1.0, -1.2, -1.4],         

    
    ("outer", 0): [-1.3, -1.5, -1.7],
    ("outer", 1): [-1.3, -1.5, -1.7],
}

def safe_one_hot_encoder():
    """Compatibilidad sklearn: sparse_output vs sparse."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def compute_robust_factor(eps_list, factors):
    """
    Política robusta maximin:
    escoge f que maximiza min_{eps} f^(1+eps).
    Devuelve: factor_opt, worst_multiplier
    """
    best_f = None
    best_worst = -np.inf
    for f in factors:
        worst_mult = min([f ** (1.0 + eps) for eps in eps_list])
       
        if (worst_mult > best_worst) or (np.isclose(worst_mult, best_worst) and best_f is not None and abs(f-1.0) < abs(best_f-1.0)):
            best_worst = worst_mult
            best_f = f
    return float(best_f), float(best_worst)

def weighted_avg(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return np.nan
    return float(np.average(values, weights=weights))

conn = sqlite3.connect(DB_NAME)
df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
conn.close()

df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
df["PULocationID"] = df["PULocationID"].astype(int)

df["log_price"] = np.log(df["avg_fare_per_mile"])

df["avg_speed_mph"] = df["avg_distance"] / (df["avg_duration_min"] / 60.0)
df = df[(df["avg_speed_mph"] > 1) & (df["avg_speed_mph"] < 60)].copy()

df["log_distance"] = np.log(df["avg_distance"].clip(lower=1e-3))
df["log_duration"] = np.log(df["avg_duration_min"].clip(lower=1e-3))
df["log_speed"] = np.log(df["avg_speed_mph"].clip(lower=1e-3))

train = df[(df["pickup_hour"] < TRAIN_END) & (df["trips"] >= MIN_TRIPS_TRAIN)].copy()
march = df[(df["pickup_hour"] >= TRAIN_END)].copy()

zh_mean = train.groupby(["PULocationID", "hour"])["log_price"].mean()
train["log_price_zh"] = train["log_price"] - train.set_index(["PULocationID", "hour"]).index.map(zh_mean)
march["log_price_zh"] = march["log_price"] - march.set_index(["PULocationID", "hour"]).index.map(zh_mean)
march["log_price_zh"] = march["log_price_zh"].fillna(0.0)

train["zone_hour"] = train["PULocationID"].astype(str) + "_" + train["hour"].astype(str)
march["zone_hour"] = march["PULocationID"].astype(str) + "_" + march["hour"].astype(str)

train_levels = set(train["zone_hour"].unique())
before = len(march)
march = march[march["zone_hour"].isin(train_levels)].copy()
print(f"Filas de Marzo removidas por zone_hour no visto en train: {before - len(march):,}")

num_features = ["log_price_zh", "log_distance", "log_duration", "log_speed", "rush_hour", "is_weekend"]
cat_features = ["zone_hour", "day_of_week"]

X_train = train[num_features + cat_features].copy()
y_train = train["trips"].astype(float)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), num_features),
        ("cat", safe_one_hot_encoder(), cat_features)
    ],
    remainder="drop"
)

model = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=POISSON_MAX_ITER, tol=POISSON_TOL)
pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
pipe.fit(X_train, y_train)

X_base = march[num_features + cat_features].copy()
X_base["log_price_zh"] = 0.0

q_base = pipe.predict(X_base)
q_base = np.clip(q_base, 1e-6, None)  

baseline_revenue = (march["avg_fare_per_mile"]) * march["avg_distance"] * q_base

zone_volume = train.groupby("PULocationID")["trips"].sum().sort_values(ascending=False)
zone_rank = zone_volume.rank(method="first", ascending=False).astype(int)

def zone_segment(pu):
    r = zone_rank.get(pu, 10**9)
    if r <= TOP_CORE_ZONES:
        return "core"
    elif r <= TOP_MID_ZONES:
        return "mid"
    else:
        return "outer"

march["zone_seg"] = march["PULocationID"].apply(zone_segment)
march["rush_seg"] = march["rush_hour"].astype(int)

segment_rows = []
seg_to_policy = {}

for seg in ["core", "mid", "outer"]:
    for rush in [0, 1]:
        eps_list = ELASTICITY_BY_SEGMENT[(seg, rush)]
        f_opt, worst_mult = compute_robust_factor(eps_list, FACTORS)
        seg_to_policy[(seg, rush)] = (f_opt, worst_mult)
        segment_rows.append({
            "zone_seg": seg,
            "rush_seg": rush,
            "eps_list": ",".join([str(e) for e in eps_list]),
            "factor_robust": f_opt,
            "worst_case_multiplier": worst_mult
        })

segments_df = pd.DataFrame(segment_rows)
segments_df.to_csv("pricing_policy_segments.csv", index=False)

print("\n=== Política robusta por segmento ===")
print(segments_df)

march["factor_robust"] = march.apply(lambda r: seg_to_policy[(r["zone_seg"], r["rush_seg"])][0], axis=1)
march["worst_mult"] = march.apply(lambda r: seg_to_policy[(r["zone_seg"], r["rush_seg"])][1], axis=1)

worst_case_revenue = baseline_revenue * march["worst_mult"].values
worst_uplift_pct = (worst_case_revenue / baseline_revenue - 1.0) * 100

march["recommend_flag"] = (
    (march["low_trips_flag"] == 0) &
    (march["trips"] >= MIN_TRIPS_FOR_RECOMMEND) &
    (march["zone_seg"] != "outer")
).astype(int)

granular = march[[
    "pickup_hour", "PULocationID", "trips",
    "avg_fare_per_mile", "avg_distance",
    "low_trips_flag", "hour", "day_of_week", "rush_hour", "is_weekend",
    "zone_seg"
]].copy()

granular["q_base_pred"] = q_base
granular["baseline_revenue"] = baseline_revenue
granular["factor_robust"] = march["factor_robust"].values
granular["price_new"] = granular["avg_fare_per_mile"] * granular["factor_robust"]
granular["worst_case_revenue_at_factor"] = worst_case_revenue
granular["worst_case_uplift_pct"] = worst_uplift_pct
granular["recommend_flag"] = march["recommend_flag"].values

g = granular[granular["recommend_flag"] == 1].copy()

zone_rows = []
for pu, grp in g.groupby("PULocationID"):
    zone_rows.append({
        "PULocationID": int(pu),
        "factor_zone_weighted": weighted_avg(grp["factor_robust"], grp["trips"]),
        "avg_worst_uplift_pct": weighted_avg(grp["worst_case_uplift_pct"], grp["trips"]),
        "total_trips": float(grp["trips"].sum()),
        "zone_seg": grp["zone_seg"].iloc[0]
    })

zone_agg = pd.DataFrame(zone_rows).sort_values("total_trips", ascending=False)

granular.to_csv("pricing_policy_granular.csv", index=False)
zone_agg.to_csv("pricing_policy_zone_agg.csv", index=False)

print("\n Exportados:")
print(" - pricing_policy_granular.csv")
print(" - pricing_policy_zone_agg.csv")
print(" - pricing_policy_segments.csv")

sim_seg_rows = []
for (seg, rush), (f_opt, worst_mult) in seg_to_policy.items():
    eps_list = ELASTICITY_BY_SEGMENT[(seg, rush)]
    for eps in eps_list:
        for f in FACTORS:
            sim_seg_rows.append({
                "zone_seg": seg,
                "rush_seg": rush,
                "eps": eps,
                "factor": float(f),
                "revenue_multiplier": float(f ** (1.0 + eps))
            })

sim_seg = pd.DataFrame(sim_seg_rows)
sim_seg.to_csv("pricing_policy_sim_segment.csv", index=False)
print(" - pricing_policy_sim_segment.csv")

coverage = granular["recommend_flag"].mean() * 100
print(f"\nCobertura de recomendaciones (marzo): {coverage:.1f}%")

dist = granular.loc[granular["recommend_flag"] == 1, "factor_robust"].value_counts(normalize=True).sort_index()
print("\nDistribución de factor_robust (solo recomendaciones):")
print(dist)

gg = granular[granular["recommend_flag"] == 1].copy()
total_base = gg["baseline_revenue"].sum()
total_worst = gg["worst_case_revenue_at_factor"].sum()
uplift_total_pct = (total_worst / total_base - 1) * 100
print(f"\nUplift robusto total (peor caso, marzo): {uplift_total_pct:.2f}%")

print("\nTop 10 zonas por volumen recomendado:")
print(zone_agg.head(10))

