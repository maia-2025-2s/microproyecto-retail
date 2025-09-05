import pandas as pd
import numpy as np
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

# 3. Definir pipeline con escalado + Lasso
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=10000))
])

# 4. Definir hiperparámetros a buscar
param_grid = {
    "lasso__alpha": np.logspace(-3, 2, 10),   # valores de 0.001 a 100
}

# 5. GridSearchCV
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# 6. Mejor modelo
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# 7. Métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mejor alpha:", grid.best_params_["lasso__alpha"])
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# 8. Guardar resultados
results = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
results.to_csv("resultados_lasso.csv", index=False)

print("Resultados guardados en resultados_lasso.csv")