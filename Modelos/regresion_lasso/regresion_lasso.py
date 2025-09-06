# Modelos/regresion_lasso_grid_mlflow.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Cargar datos
data = pd.read_csv("../EDA/data/train.csv")

# ---- EXTRA: features de fecha ----
data["date"] = pd.to_datetime(data["date"])
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["dayofweek"] = data["date"].dt.dayofweek
data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
data = data.drop(columns=["date"])

X = data.drop(columns=["sales"])
y = data["sales"]

# 2. Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Definir pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=10000))
])

# 4. Definir hiperparámetros
param_grid = {
    "lasso__alpha": np.logspace(-3, 2, 10),
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

# 5. Experimento MLflow
mlflow.set_experiment("Regresion_Lasso")

with mlflow.start_run():
    grid.fit(X_train, y_train)

    # Mejor modelo
    best_alpha = grid.best_params_["lasso__alpha"]
    best_model = grid.best_estimator_

    # Predicciones
    y_pred = best_model.predict(X_test)

    # Métricas
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ---- MLflow logging ----
    mlflow.log_param("alpha", best_alpha)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)

    # Guardar modelo
    mlflow.sklearn.log_model(best_model, "lasso_model")

    print("Mejor alpha:", best_alpha)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")