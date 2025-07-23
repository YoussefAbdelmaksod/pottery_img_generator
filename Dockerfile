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
COPY app.py .
COPY pipeline/ ./pipeline/
COPY lora-output/ ./lora-output/
COPY adapter_config.json .

# Create directories
RUN mkdir -p /tmp

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Set environment variables to use /tmp for caching (writable in Hugging Face Spaces)
ENV PYTHONPATH=/app
ENV TORCH_HOME=/tmp/torch_cache
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HUB_CACHE=/tmp/hf_hub_cache

# Create cache directories in /tmp with proper permissions
RUN mkdir -p /tmp/torch_cache /tmp/hf_cache /tmp/transformers_cache /tmp/hf_hub_cache && \
    chmod -R 777 /tmp

# Run the application
CMD ["python", "app.py"]
