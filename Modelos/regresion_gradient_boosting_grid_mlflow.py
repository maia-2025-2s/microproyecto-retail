# Modelos/regresion_gradient_boosting_grid_mlflow.py
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- 1) Cargar datos (path robusto) ----------
candidates = ["../EDA/data/train.csv", "../data/train.csv"]
data_path = next((p for p in candidates if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError(f"No encontré train.csv en: {candidates}")

data = pd.read_csv(data_path)

# ---------- EXTRA: features rápidas ----------
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values(["store", "item", "date"])

# Partes de fecha
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["dayofweek"] = data["date"].dt.dayofweek
data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

# Lags/rolling sencillos por (store, item)
data["lag_1"]  = data.groupby(["store","item"])["sales"].shift(1)
data["lag_7"]  = data.groupby(["store","item"])["sales"].shift(7)
data["rmean_7"] = data.groupby(["store","item"])["sales"].shift(1).rolling(7).mean()

# Quitar filas con NaNs creados por lags/rolling y descartar 'date'
data = data.dropna().drop(columns=["date"]).reset_index(drop=True)

X = data.drop(columns=["sales"])
y = data["sales"]

# ---------- 2) Split 80/20 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 3) Modelo + grid compacto ----------
model = GradientBoostingRegressor(random_state=42)
param_grid = {
    "n_estimators": [200, 400],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3],
    "subsample": [0.7, 1.0],
}

grid = GridSearchCV(
    model, param_grid,
    cv=3, scoring="neg_mean_squared_error",
    n_jobs=-1, verbose=0
)

# ---------- 4) MLflow ----------
mlflow.set_experiment("Regresion_GradientBoosting")

with mlflow.start_run(run_name="GB_Felipe"):
    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_model  = grid.best_estimator_

    # Predicciones + métricas
    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    # Log params/metrics
    for k,v in best_params.items(): mlflow.log_param(k, v)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE",  mae)
    mlflow.log_metric("R2",   r2)

    # Artefactos: features y modelo
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/features.txt", "w") as f:
        f.write("\n".join(map(str, X.columns.tolist())))
    mlflow.log_artifact("artifacts/features.txt")

    mlflow.sklearn.log_model(best_model, artifact_path="gb_model")

    # Tags útiles para la UI
    mlflow.set_tag("author", "Pablo Felipe Rengifo Montilla")
    mlflow.set_tag("branch", "regresion-lineal")
    mlflow.set_tag("dataset", "demand-forecasting-kernels-only")
    mlflow.set_tag("script", "Modelos/regresion_gradient_boosting_grid_mlflow.py")

    print("Mejores hiperparámetros:", best_params)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")