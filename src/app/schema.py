from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import pandas as pd


BASE_PATH = Path(__file__).parents[2].resolve()
MODEL_PATH = BASE_PATH / "models" / "artifacts" / "modelo_final.joblib"
UMBRAL = 0.5
 
 
modelo = joblib.load(MODEL_PATH)
COLUMNAS = list(modelo.feature_names_in_)
 
app = FastAPI(title="API de Scoring Crediticio")


class SolicitudCredito(BaseModel): 
    edad: int = Field(ge=18, le=100, description="Edad del solicitante en anios")
    personas_a_cargo: int = Field(ge=0, description="Numero de personas a cargo")
    ingreso_mensual: float = Field(ge=0, description="Ingreso mensual del solicitante")
    monto_solicitado: float = Field(gt=0, description="Monto del prestamo solicitado")
    plazo_meses: int = Field(gt=0, description="Plazo del credito en meses")
    cuota_estimada: float = Field(gt=0, description="Cuota mensual estimada")
    score_crediticio: float = Field(description="Score de historial crediticio")
    dti_previo: float = Field(ge=0, description="Relacion deuda/ingreso antes del nuevo prestamo")
    dti_total_proyectado: float = Field(ge=0, description="Relacion deuda/ingreso proyectada con el nuevo prestamo")
    

@app.get("/health")
def health():
    return {"status": "ok", "n_variables": len(COLUMNAS)}


 
@app.post("/predict")
def predict(solicitud: SolicitudCredito):
    df = pd.DataFrame([solicitud.model_dump()])[COLUMNAS]
    prob = float(modelo.predict_proba(df)[0, 1])
 
    return {
        "aprobado": bool(prob >= UMBRAL),
        "probabilidad": round(prob, 4),
        "umbral": UMBRAL,
    }
 