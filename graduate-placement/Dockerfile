FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for xgboost and others
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --default-timeout=100 --retries 10 --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
