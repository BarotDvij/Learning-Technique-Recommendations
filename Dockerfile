# syntax=docker/dockerfile:1.6
# Slim, single-stage image for the Streamlit app.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application code
COPY app.py ./
COPY src ./src
COPY .streamlit ./.streamlit

EXPOSE 8501

# Use Streamlit's built-in health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status == 200 else 1)" \
    || exit 1

CMD ["streamlit", "run", "app.py"]
