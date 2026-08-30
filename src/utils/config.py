import yaml
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


REGISTRO = {
    'LogistcRegression': LogisticRegression, 
    'RandomForestClassifier': RandomForestClassifier, 
    'XGBClassifier': XGBClassifier, 
    'KNeighborsClassifier': KNeighborsClassifier, 
    'SVC': SVC, 
}

def read_config() -> dict:
    
    BASE_PATH = Path(__file__).parents[2].resolve()
    CFG_PATH = BASE_PATH / "config"/ "config.yaml"
    
    with open(CFG_PATH, mode='r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    return cfg


def build_model() -> dict:
    
    cfg = read_config()
    
    models = {}
    for name,spec in cfg['base_models'].items():
        cls = REGISTRO[spec['class']]
        models[name] = cls(**spec.get('params', {}))
    return models
