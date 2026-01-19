# 02b_eda_hourly_zone_panel.py
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

DB_NAME = "taxi_pricing_v2.db"
TABLE = "hourly_zone_panel"

conn = sqlite3.connect(DB_NAME)
panel = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
conn.close()

panel["pickup_hour"] = pd.to_datetime(panel["pickup_hour"])

# 1) Distribución avg_fare_per_mile
plt.figure(figsize=(8,5))
panel["avg_fare_per_mile"].plot(kind="hist", bins=60)
plt.title("Distribución de avg_fare_per_mile (hora×zona)")
plt.xlabel("avg_fare_per_mile")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("dist_avg_fare_per_mile.png", dpi=150)
plt.close()

# 2) Distribución trips
plt.figure(figsize=(8,5))
panel["trips"].plot(kind="hist", bins=60, log=True)
plt.title("Distribución de trips (hora×zona) [escala log en Y]")
plt.xlabel("trips")
plt.ylabel("Frecuencia (log)")
plt.tight_layout()
plt.savefig("dist_trips_hour_zone.png", dpi=150)
plt.close()

# 3) Heatmap hora×día (global): suma de trips
tmp = panel.groupby(["day_of_week","hour"])["trips"].sum().reset_index()
pivot = tmp.pivot(index="day_of_week", columns="hour", values="trips")

plt.figure(figsize=(12,4))
plt.imshow(pivot.values, aspect="auto")
plt.colorbar(label="trips (suma)")
plt.title("Heatmap demanda total: día de semana × hora")
plt.yticks(range(7), ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"])
plt.xticks(range(24), range(24))
plt.xlabel("Hora del día")
plt.ylabel("Día de semana")
plt.tight_layout()
plt.savefig("heatmap_trips_dow_hour.png", dpi=150)
plt.close()

# 4) Top zonas por volumen
top_zones = panel.groupby("PULocationID")["trips"].sum().sort_values(ascending=False).head(15)
plt.figure(figsize=(10,5))
top_zones.sort_values().plot(kind="barh")
plt.title("Top 15 zonas por demanda total (trips)")
plt.xlabel("trips (suma)")
plt.ylabel("PULocationID")
plt.tight_layout()
plt.savefig("top_zones_by_trips.png", dpi=150)
plt.close()

print("Gráficos guardados: dist_avg_fare_per_mile.png, dist_trips_hour_zone.png, heatmap_trips_dow_hour.png, top_zones_by_trips.png")

