from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import joblib
import io

app = FastAPI()

# Permite que el frontend pueda hacer peticiones a esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir esto al dominio del frontend si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Rutas de archivos locales
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Modelos", "modelo_entrenado.pkl")



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Modelos", "random_forest", "model.pkl")

try:
    modelo = joblib.load(MODEL_PATH)
    print(f"✅ Modelo cargado desde {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ No se pudo cargar el modelo: {e}")
    modelo = None

# ========================
# Endpoints
# ========================

@app.get("/")
def root():
    return {"message": "API de predicción de demanda lista."}

@app.get("/preview")
def preview():
    try:
        df = pd.read_csv(TRAIN_PATH)
        return df.head(10).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer CSV: {e}")

@app.get("/predict")
def predict_from_query(store: int, item: int):
    if not modelo:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    try:
        fechas = pd.date_range(start="2018-01-01", periods=14)

        df = pd.DataFrame({
            "store": [store] * 14,
            "item": [item] * 14,
            "day": fechas.day,
            "month": fechas.month,
            "dow": fechas.dayofweek,  # 👈 mismo nombre que en el entrenamiento
        })

        yhat = modelo.predict(df)

        resultados = []
        for i in range(len(fechas)):
            resultados.append({
                "date": fechas[i].strftime("%Y-%m-%d"),
                "store": store,
                "item": item,
                "yhat": round(float(yhat[i]), 2),
            })

        return resultados

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {e}")
    

@app.get("/options")
def get_options():
    try:
        df = pd.read_csv(TRAIN_PATH)
        stores = sorted(df["store"].unique().tolist())
        items = sorted(df["item"].unique().tolist())
        return {"stores": stores, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener opciones: {e}")


