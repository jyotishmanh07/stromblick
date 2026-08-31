FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY data ./data
RUN pip install --no-cache-dir .

EXPOSE 8000 8501
CMD ["uvicorn", "energy_forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
