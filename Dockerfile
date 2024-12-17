FROM pytorch/pytorch:2.0.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy training code
COPY trainer /app/trainer
COPY setup.py /app/

# Install Python dependencies
RUN pip install -e .

# Set the entrypoint
ENTRYPOINT ["python", "-m", "trainer.task"]