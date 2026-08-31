'''Se definen los espacios de busqueda'''

def logreg_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
    }

def svm_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
    }

def random_forest_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

def xgboost_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }

def knn_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
        "p": trial.suggest_categorical("p", [1, 2]),
    }


SEARCH_SPACES = {
    "logreg": logreg_space,
    "svm": svm_space,
    "random_forest": random_forest_space,
    "xgboost": xgboost_space,
    "knn": knn_space,
}
