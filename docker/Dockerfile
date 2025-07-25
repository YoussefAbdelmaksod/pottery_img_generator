FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY docker/app.py .
COPY pipeline/ ./pipeline/
COPY lora-output/ ./lora-output/
COPY adapter_config.json .

# Create directories
RUN mkdir -p /tmp

# Expose port
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/torch_cache
ENV HF_HOME=/app/hf_cache

# Create cache directories
RUN mkdir -p /app/torch_cache /app/hf_cache

# Run the application
CMD ["python", "app.py"]
