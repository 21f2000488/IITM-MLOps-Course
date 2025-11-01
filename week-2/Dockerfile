FROM python:3.10-slim

WORKDIR /app

# Copy requirements and app
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY iris_fastapi.py ./

EXPOSE 8200

CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"]
