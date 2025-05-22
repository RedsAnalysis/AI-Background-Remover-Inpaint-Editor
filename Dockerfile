# Use an official Python 3.12 runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install git, which is needed to clone the repository
# Also install system dependencies that might be needed by OpenCV or Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# For simplicity, hardcoding the URL here:
RUN git clone https://github.com/RedsAnalysis/AI-Background-Remover-Inpaint-Editor.git .
# The "." at the end clones the repo content directly into the current WORKDIR (/app)

# Install Python dependencies from the cloned requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available (Gradio's default port)
EXPOSE 7860

# Command to run the application
# Ensure your app.py's iface.launch() is configured to listen on 0.0.0.0
# or that GRADIO_SERVER_NAME/PORT are set in docker-compose.yml
CMD ["python3", "-u", "app.py"]