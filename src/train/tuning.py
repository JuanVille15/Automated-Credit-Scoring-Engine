'''Optimizacion de hiperparametros (Optuna) para un modelo dado'''

import pandas as pd
import optuna
import json
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

from pathlib import Path
import joblib

from src.utils.config import REGISTRO, read_config
from src.utils.search_spaces import SEARCH_SPACES


def _optimize(model_name: str, x, y, cv, n_trials, scoring: str = "roc_auc"):

    cfg = read_config()
    spec = cfg["base_models"][model_name]
    cls = REGISTRO[spec["class"]]
    fixed_params = spec.get("params", {})
    
    space_fn = SEARCH_SPACES[model_name]

    def objective(trial):
        tuned_params = space_fn(trial)
        modelo = cls(**{**fixed_params, **tuned_params})
        pipe = Pipeline([("scaler", StandardScaler()), ("model", modelo)])
        scores = cross_val_score(pipe, x, y, cv=cv, scoring=scoring)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    return {'model': model_name, 'parametros':study.best_params, 'roc-auc': study.best_value}

def train_best_model(x:pd.DataFrame, y:pd.DataFrame, cv, n_trials:int=100):
    
    BASE_PATH = Path(__file__).parents[2].resolve()
    LEADER_BOARD_PATH = BASE_PATH / "models" / "leaderboard.json"
    PATH_OUT = BASE_PATH / "models" / "tuning.json"
    
    # --- Se cargan los campeones --- #
    with open(LEADER_BOARD_PATH, 'r', encoding='utf-8') as file:
        leader_board = json.load(file)
    
    top_3 = leader_board[:3]
    
    # --- Se guardan los resultados --- #
    result = []
    for model in top_3:
        model_name = model.get('model', [])
        print(f'Optimizando: {model_name}...')
        r =_optimize(model_name=model_name, 
                  x=x, 
                  y=y,
                  cv=cv, 
                  n_trials=n_trials, 
                  )
        print(f'{model_name} optimizado con: {n_trials} intentos')
        result.append(r)
    
    # --- Se escribe los resultados en models --- #
    tuning_final = pd.DataFrame(result).sort_values(by='roc-auc', ascending=False)
    tuning_final.to_json(PATH_OUT, orient='records', indent=2)
    
    # --- Se entrena el mejor modelo --- #
    best_model_name = tuning_final['model'].iloc[0]
    
    cfg = read_config()
    spec = cfg['base_models'][best_model_name]
    cls = REGISTRO[spec['class']]
    best_params = tuning_final['parametros'].iloc[0]
    modelo_final = Pipeline([
        ("scaler", StandardScaler()),
        ("model", cls(**{**spec.get('params', {}), **best_params})),
    ])
    modelo_final.fit(x, y)
    
    return modelo_final

def evaluate(modelo_final, x_test, y_test) -> dict:

    BASE_PATH = Path(__file__).parents[2].resolve()

    y_pred = modelo_final.predict(x_test)
    if hasattr(modelo_final, "predict_proba"):
        y_score = modelo_final.predict_proba(x_test)[:, 1]
    else:
        y_score = modelo_final.decision_function(x_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_score),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    OUT_PATH = BASE_PATH / "models" / "final_metrics.json"
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    ARTIFACT_PATH = BASE_PATH / "models" / "artifacts" / "modelo_final.joblib"
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_final, ARTIFACT_PATH)

    return metrics