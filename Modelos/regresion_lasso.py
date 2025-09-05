import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === 1. Cargar tus datos (ya limpios) ===
# Supongamos que tienes un DataFrame 'df' con X (features) e y (target)
X = df.drop("target", axis=1)
y = df["target"]

# === 2. División en entrenamiento y prueba ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Definir modelos y grids ===
ridge = Ridge()
lasso = Lasso()

param_grid = {"alpha": np.logspace(-4, 4, 50)}  # 50 valores de 1e-4 a 1e4

ridge_search = GridSearchCV(ridge, param_grid, cv=5, scoring="r2", n_jobs=-1)
lasso_search = GridSearchCV(lasso, param_grid, cv=5, scoring="r2", n_jobs=-1)

# === 4. Entrenamiento con búsqueda de hiperparámetros ===
ridge_search.fit(X_train, y_train)
lasso_search.fit(X_train, y_train)

# === 5. Evaluación en test ===
def evaluar(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return {"R2": r2, "MAE": mae, "RMSE": rmse}

resultados = {
    "Ridge": {
        "Mejor alpha": ridge_search.best_params_["alpha"],
        **evaluar(ridge_search.best_estimator_, X_test, y_test),
    },
    "Lasso": {
        "Mejor alpha": lasso_search.best_params_["alpha"],
        **evaluar(lasso_search.best_estimator_, X_test, y_test),
    },
}

# Mostrar resultados
for modelo, metrics in resultados.items():
    print(f"\nModelo: {modelo}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")