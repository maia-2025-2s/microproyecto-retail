# -*- coding: utf-8 -*-

#Importa pandas
import pandas as pd

# Cargar los datos
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Convertir la columna 'date' a tipo datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])


# Extraer características de fecha
train_df['year'] = train_df['date'].dt.year
train_df['month'] = train_df['date'].dt.month
train_df['dayofweek'] = train_df['date'].dt.dayofweek
train_df['dayofyear'] = train_df['date'].dt.dayofyear

test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['dayofweek'] = test_df['date'].dt.dayofweek
test_df['dayofyear'] = test_df['date'].dt.dayofyear


# Eliminar la columna 'date' ya que no es necesaria para el modelo
train_df = train_df.drop('date', axis=1)
test_df = test_df.drop('date', axis=1)

from sklearn.model_selection import train_test_split

# Dividir los datos en características (X) y objetivo (y)
X = train_df.drop('sales', axis=1)
y = train_df['sales']

# Dividir los datos en conjuntos de entrenamiento y validación (80% - 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)

# mlflow
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import Ridge

mlflow.set_tracking_uri('http://34.205.7.29:8050')

experiment = mlflow.set_experiment("Regresion Lineal Ridge")
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # defina los parámetros del modelo
    alpha = 1.0  # Parámetro de regularización para Ridge Regression 

    # Cree el modelo con los parámetros definidos y entrénelo
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Predict on the validation set
    predictions = model.predict(X_val)

    # Registre los parámetros
    mlflow.log_param("alpha", alpha)
  
    # Registre el modelo
    mlflow.sklearn.log_model(model, "regresion_lineal_ridge_model")
    
    # Evaluate the model
    mse = mean_squared_error(y_val, predictions)
    mlflow.log_metric("mse", mse)


print(f'Mean Squared Error (MSE): {mse}')

# Predict on the test set
test_predictions = model.predict(test_df.drop('id', axis=1))

# Create a DataFrame with predictions
predictions_df = pd.DataFrame({'sales_prediction': test_predictions})

# Merge test_df with predictions_df based on index
output_df = test_df.copy()
output_df['sales_prediction'] = predictions_df['sales_prediction']

# Save the results to a CSV file
output_df.to_csv('predicciones.csv', index=False)

print("Predictions saved to predicciones.csv")
