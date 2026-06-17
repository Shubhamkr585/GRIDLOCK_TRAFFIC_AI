FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port for FastAPI (to be implemented later)
EXPOSE 8000

# Default command to run the FastAPI server instead of training the model
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
