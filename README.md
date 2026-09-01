# Automated Credit Scoring Engine

API de scoring crediticio desarrollada con FastAPI y un modelo de Machine Learning serializado con `joblib`.



## Construir la imagen Docker

Desde la raíz del proyecto:

```bash
docker build -t credit-scoring-api .
```

## Ejecutar el contenedor

```bash
docker run --rm --name credit-scoring-container -p 8000:8000 credit-scoring-api
```

La API estará disponible en:

http://localhost:8000
