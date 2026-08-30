import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# 1. CARGAR DATASET
df = pd.read_parquet("src\credit\datos_credito.parquet")

# 2. SEPARAR VARIABLES PREDICTORAS Y VARIABLE RESPUESTA
TARGET = "aprobado"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 3. DIVIDIR EN TRAIN Y VALIDACIÓN
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

# 4. CREAR PIPELINE
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 5. ENTRENAR
model.fit(X_train, y_train)


# 6. VALIDAR
y_pred = model.predict(X_val)

# Probabilidad de la clase 1 = APROBADO
y_prob = model.predict_proba(X_val)[:, 1]


# 7. MÉTRICAS
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)

print("RESULTADOS DE VALIDACIÓN")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nMatriz de confusión:")
print(confusion_matrix(y_val, y_pred))

print("\nReporte de clasificación:")
print(classification_report(
    y_val,
    y_pred,
    target_names=["No aprobado", "Aprobado"],
    zero_division=0
))


# 8. GUARDAR MODELO
MODEL_PATH = "models/modelo.joblib"
joblib.dump(model, MODEL_PATH)