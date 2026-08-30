from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from src.utils.config import build_model

import pandas as pd
import numpy as np
from pathlib import Path

def train_model(
    model:str, 
    clase:BaseEstimator, 
    x:pd.DataFrame, 
    y:pd.DataFrame, 
    cv: StratifiedKFold, 
    scoring:list[str]
) -> dict:
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", clase),
    ])
    scores = cross_validate(pipe, x, y, cv=cv, scoring=scoring)

    return {'model':model, **{m:scores[f'test_{m}'].mean() for m in scoring}}

def run_leaderbord(df:pd.DataFrame):
    
    models = build_model()
    x_train, x_test, y_train, y_test = train_test_split(
    df.drop(columns="aprobado"), df["aprobado"],
    test_size=0.2, random_state=42, stratify=df["aprobado"],
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["roc_auc", "recall", "precision", "f1"]
    BASE_PATH = Path(__file__).parents[2].resolve()
    OUT_PATH = BASE_PATH / "models" / "leaderboard.json"
    
    print('Corriendo Champion Select --- No optimizado')
    results = []
    for name, model in models.items():
        print(f'Entrenando: {name}...')
        resultado = train_model(model=name, clase=model, x=x_train, y=y_train, cv=cv, scoring=scoring)
        results.append(resultado)
        print(f'{name} entrenado correctamente...')
        
    leaderbord = pd.DataFrame(results).sort_values('roc_auc', ascending=False).to_json(OUT_PATH, orient='records', indent=2)
    

if __name__ == '__main__':
    BASE_PATH = Path(__file__).parents[2].resolve()
    df = pd.read_parquet(BASE_PATH / "data" / "datos_credito.parquet")
    run_leaderbord(df)