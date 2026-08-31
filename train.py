"""Entrypoint: corre el pipeline completo de entrenamiento.

    1. Carga los datos y hace el UNICO split train/test del proyecto.
    2. Champion select: compara los modelos base, sin tunear (leaderboard.json).
    3. Tuning: optimiza con Optuna el top-3 del leaderboard, elige y reentrena al ganador.
    4. Evaluacion final, una sola vez, sobre el test set.
"""

from sklearn.model_selection import StratifiedKFold, train_test_split

from src.utils.data import load_data
from src.train.leaderbord import run_leaderbord
from src.train.tuning import train_best_model, evaluate

TARGET = "aprobado"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 100


def main():
    df = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=TARGET), df[TARGET],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET],
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # 1. Champion select
    run_leaderbord(x_train, y_train, cv=cv)

    # 2. Tuning del top-3 + entrenamiento del ganador
    modelo_final = train_best_model(x_train, y_train, cv=cv, n_trials=N_TRIALS)

    # 3. Evaluacion honesta, una sola vez, en el test set
    metrics = evaluate(modelo_final, x_test, y_test)
    print("Metricas finales:", metrics)


if __name__ == "__main__":
    main()
