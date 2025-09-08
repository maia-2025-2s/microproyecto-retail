# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

#cambio para probar git

import mlflow
import mlflow.sklearn

# ==============================
# 1. Cargar datos
# ==============================
train_df = pd.read_csv("train.csv", parse_dates=["date"])
test_df = pd.read_csv("test.csv", parse_dates=["date"])

# ==============================
# 2. Features temporales
# ==============================
for df in [train_df, test_df]:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek

# ==============================
# 3. Crear lags y rolling means
# ==============================
train_df = train_df.sort_values(["store", "item", "date"])

for lag in [1, 7, 30]:
    train_df[f"lag_{lag}"] = train_df.groupby(["store", "item"])["sales"].shift(lag)

train_df["rolling_mean_7"] = train_df.groupby(["store", "item"])["sales"].shift(1).rolling(7).mean()
train_df["rolling_mean_30"] = train_df.groupby(["store", "item"])["sales"].shift(1).rolling(30).mean()

# Eliminar NaN generados por los lags
train_df = train_df.dropna()

# ==============================
# 4. Variables categóricas
# ==============================
for df in [train_df, test_df]:
    df["store"] = df["store"].astype("category")
    df["item"] = df["item"].astype("category")

# One-hot encoding
X = pd.get_dummies(train_df.drop(["sales", "date"], axis=1), drop_first=True)
y = train_df["sales"]

X_test_final = pd.get_dummies(test_df.drop(["id", "date"], axis=1), drop_first=True)

# Alinear columnas entre train y test
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

# ==============================
# 5. División temporal (último 20% como validación)
# ==============================
val_size = int(len(train_df) * 0.2)
X_train, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
y_train, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

# ==============================
# 6. Configuración de MLflow
# ==============================
mlflow.set_tracking_uri("http://34.205.7.29:8050")
experiment = mlflow.set_experiment("Regresion Lineal Ridge")

alpha = 50  # parámetro fijo

with mlflow.start_run(experiment_id=experiment.experiment_id):
    # Modelo en pipeline con escalado
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

    # Entrenar
    pipeline.fit(X_train, y_train)

    # ==============================
    # 7. Evaluación en validación
    # ==============================
    y_val_pred = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_val_pred)

    print(f"Alpha fijo: {alpha}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # ==============================
    # 8. Registro en MLflow
    # ==============================
    mlflow.log_param("alpha", alpha)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    mlflow.sklearn.log_model(pipeline, "regresion_lineal_ridge_model")

    # ==============================
    # 9. Entrenar con todo el dataset y predecir test
    # ==============================
    pipeline.fit(X, y)
    test_predictions = pipeline.predict(X_test_final)

    output_df = test_df.copy()
    output_df["sales_prediction"] = test_predictions
    output_df.to_csv("predicciones.csv", index=False)

    print("Predicciones guardadas en predicciones.csv ✅")
