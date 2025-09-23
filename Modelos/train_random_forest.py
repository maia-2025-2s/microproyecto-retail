import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

DATA_PATH = "../data/raw/train.csv"

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["dow"] = df["date"].dt.dayofweek

X = df[["store", "item", "day", "month", "dow"]]
y = df["sales"]

model = RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

out_dir = "random_forest"
os.makedirs(out_dir, exist_ok=True)

joblib.dump(model, os.path.join(out_dir, "model.pkl"))
print(" Modelo guardado en", os.path.join(out_dir, "model.pkl"))
