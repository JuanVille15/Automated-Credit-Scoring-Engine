from pathlib import Path
 
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
from src.train.tuning import evaluate
from src.utils.config import REGISTRO, read_config
 
BASE_PATH = Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH / "data" / "datos_credito.parquet"
TUNING_PATH = BASE_PATH / "models" / "tuning.json"
ARTIFACT_PATH = BASE_PATH / "models" / "artifacts" / "modelo_final.joblib"
 
 
def construir_pipeline(spec: dict, best_params: dict) -> Pipeline:
    """Arma el pipeline con los parametros fijos del YAML mas los optimizados."""
    fixed = dict(spec.get("params", {}))
 
    # El SVM necesita probability=True para exponer predict_proba en la API.
    if spec["class"] == "SVC":
        fixed["probability"] = True
 
    cls = REGISTRO[spec["class"]]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", cls(**{**fixed, **best_params})),
    ])
 
 
def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    x = df.drop(columns="aprobado")
    y = df["aprobado"]
 

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )
 
    tuning = pd.read_json(TUNING_PATH).sort_values("roc-auc", ascending=False)
    best = tuning.iloc[0]
    spec = read_config()["base_models"][best["model"]]
 
    print(f"Modelo ganador: {best['model']} ({spec['class']})")
    print(f"Parametros: {best['parametros']}")
 
    # Entrenar con train y evaluar contra test
    print("\nEntrenando sobre train para evaluar...")
    modelo_eval = construir_pipeline(spec, best["parametros"])
    modelo_eval.fit(x_train, y_train)
 
    metricas = evaluate(modelo_eval, x_test, y_test)
    print("\nMetricas sobre el conjunto de prueba:")
    for k, v in metricas.items():
        print(f"  {k}: {v}")
 
    # Reentrenar con todos los datos
    print("\nReentrenando sobre el 100% de los datos...")
    modelo_prod = construir_pipeline(spec, best["parametros"])
    modelo_prod.fit(x, y)
 
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_prod, ARTIFACT_PATH)
 
    print(f"\nModelo guardado en: {ARTIFACT_PATH}")
    print(f"predict_proba disponible: {hasattr(modelo_prod, 'predict_proba')}")
 
 
if __name__ == "__main__":
    main()