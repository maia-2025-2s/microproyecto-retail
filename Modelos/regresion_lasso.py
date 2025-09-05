# Modelos/regresion_lasso.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Cargar datos (por ahora desde EDA)
data = pd.read_csv("../EDA/data/train.csv")

# convertir a datetime
data["date"] = pd.to_datetime(data["date"])

# extraer características
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["dayofweek"] = data["date"].dt.dayofweek  # 0 = lunes, 6 = domingo
data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
data["dayofyear"] = data["date"].dt.dayofyear

# eliminar la columna de fecha cruda (no sirve como string)
data = data.drop(columns=["date"])

X = data.drop(columns=["sales"])   # variable dependiente
y = data["sales"]                  # nombre real de la columna objetivo

# 2. Separar en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Definir modelo con validación cruzada para alpha
lasso = LassoCV(cv=5, random_state=42)

# 4. Entrenar
lasso.fit(X_train, y_train)

# 5. Predicciones
y_pred = lasso.predict(X_test)

# 6. Métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mejor alpha:", lasso.alpha_)
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# 7. Guardar resultados en CSV
results = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
results.to_csv("resultados_lasso.csv", index=False)

print("Resultados guardados en resultados_lasso.csv")