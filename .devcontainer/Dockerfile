FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Add Node 18 and ffmpeg
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get update && \
    apt-get install -y nodejs ffmpeg

# Install common Python libs now to cache them
RUN pip install --no-cache-dir pandas pbpstats xgboost scikit-learn streamlit fastapi uvicorn
