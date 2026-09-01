FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY src ./src
COPY models ./models
CMD ["uv", "run", "uvicorn", "src.app.schema:app", "--host", "0.0.0.0", "--port", "8000"]
