# Use an official Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Reduce memory usage for PyTorch/libraries
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV and tools
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your project files
COPY . /app/

# Make the entrypoint script executable
RUN chmod +x /app/docker-entrypoint.sh

# Explicitly tell Docker which port we're using
EXPOSE 8080

# Add a health check to help containers and orchestrators know when the app is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use our custom entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run the application using Gunicorn production server
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "120", "application:app"]
