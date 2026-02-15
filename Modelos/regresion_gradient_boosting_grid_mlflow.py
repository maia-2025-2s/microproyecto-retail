# Modelos/regresion_gradient_boosting_grid_mlflow.py
import os
import time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform

# ----------------- 0) Perillas de velocidad -----------------
FAST = os.getenv("FAST_RUN", "0") == "1"                    # set FAST_RUN=1 para modo rápido
DATA_FRACTION = float(os.getenv("DATA_FRACTION", "1.0"))    # e.g., 0.5 = 50% de (store,item)

# ---------- 1) Cargar datos (path robusto) ----------
candidates = ["../EDA/data/train.csv", "../data/train.csv"]
data_path = next((p for p in candidates if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError(f"No encontré train.csv en: {candidates}")
data = pd.read_csv(data_path)

# ---------- EXTRA: features rápidas ----------
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values(["store", "item", "date"]).reset_index(drop=True)

# Partes de fecha
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["dayofweek"] = data["date"].dt.dayofweek
data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)

# Lags/rolling sencillos por (store, item)
data["lag_1"]   = data.groupby(["store","item"])["sales"].shift(1)
data["lag_7"]   = data.groupby(["store","item"])["sales"].shift(7)
data["rmean_7"] = data.groupby(["store","item"])["sales"].shift(1).rolling(7).mean()

# (Opcional) acelerar usando un subconjunto de pares store-item (sin romper lags)
if DATA_FRACTION < 1.0:
    pairs = data[["store","item"]].drop_duplicates()
    keep_n = max(1, int(len(pairs) * DATA_FRACTION))
    keep_pairs = pairs.sample(keep_n, random_state=42)
    data = data.merge(keep_pairs.assign(_keep=1), on=["store","item"], how="inner")

# Quitar filas con NaNs creados por lags/rolling y descartar 'date'
data = data.dropna().drop(columns=["date"]).reset_index(drop=True)

X = data.drop(columns=["sales"])
y = data["sales"]

# ---------- 2) Split 80/20 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 3) Modelo + búsqueda aleatoria (rápida) ----------
model = GradientBoostingRegressor(random_state=42)

# espacios de búsqueda (distribuciones continuas/discretas)
param_distributions = {
    "n_estimators": randint(120, 401),          # 120–400
    "learning_rate": uniform(0.03, 0.12),       # ~0.03–0.15
    "max_depth": randint(2, 4),                 # {2,3}
    "subsample": uniform(0.7, 0.3),             # ~0.7–1.0
}

n_iter = 8 if FAST else 16                      # menos iteraciones en modo rápido
cv_folds = 2                                    # cv más pequeño → más veloz

grid = RandomizedSearchCV(
    model,
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=cv_folds,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# ---------- 4) MLflow ----------
mlflow.set_experiment("Regresion_GradientBoosting")
start = time.perf_counter()

with mlflow.start_run(run_name="GB_Felipe_Fast" if FAST else "GB_Felipe"):
    # tags útiles antes de entrenar (aparecen en la UI aunque el fit tarde)
    mlflow.set_tag("author", "Pablo Felipe Rengifo Montilla")
    mlflow.set_tag("branch", "regresion-lineal")
    mlflow.set_tag("dataset", "demand-forecasting-kernels-only")
    mlflow.set_tag("script", "Modelos/regresion_gradient_boosting_grid_mlflow.py")
    mlflow.set_tag("fast_run", str(FAST))
    mlflow.set_tag("data_fraction", str(DATA_FRACTION))
    mlflow.log_param("cv", cv_folds)
    mlflow.log_param("n_iter", n_iter)

    grid.fit(X_train, y_train)
    best_params = grid.best_params_
    best_model  = grid.best_estimator_

    # Predicciones + métricas
    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    dur  = float(time.perf_counter() - start)

    # Log params/metrics
    for k,v in best_params.items():
        # recorta subsample a [0,1] por si la distr. uniformo lo deja >1.0 por bordes
        mlflow.log_param(k, min(1.0, v) if k=="subsample" else v)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE",  mae)
    mlflow.log_metric("R2",   r2)
    mlflow.log_metric("fit_seconds", dur)

    # Artefactos: features y modelo
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/features.txt", "w") as f:
        f.write("\n".join(map(str, X.columns.tolist())))
    mlflow.log_artifact("artifacts/features.txt")

    mlflow.sklearn.log_model(best_model, artifact_path="gb_model")

    print("Mejores hiperparámetros:", best_params)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Tiempo total: {dur:.1f}s")
