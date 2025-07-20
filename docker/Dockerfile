FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your LoRA weights
COPY lora_weights/ ./lora_weights/
COPY app.py .

# Create directories
RUN mkdir -p /tmp

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/torch_cache
ENV HF_HOME=/app/hf_cache

# Create cache directories
RUN mkdir -p /app/torch_cache /app/hf_cache

# Run the application
CMD ["python", "app.py"]
