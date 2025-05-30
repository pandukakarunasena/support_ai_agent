# ─── Stage 1: Build Python deps ────────────────────────────────────────────────
FROM python:3.10-slim AS builder
WORKDIR /app

# Install build tools, then clean up apt caches
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements & install into /install
COPY requirements.txt .
RUN pip install \
      --no-cache-dir \
      --prefix=/install \
      -r requirements.txt

# Strip out tests, docs, pycaches, and unneeded symbols
RUN find /install/lib/python3.10/site-packages \
      -type d \( -name "tests" -o -name "__pycache__" -o -name "docs" -o -name "doc" \) -prune \
      -exec rm -rf {} + \
    && find /install/lib/python3.10/site-packages -type f -name '*.so' \
      -exec strip --strip-unneeded {} +


# ─── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.10-slim
WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Copy the trimmed Python packages
COPY --from=builder /install /usr/local

# Copy app code
COPY . .

# Prepare logs dir & HF cache volume
RUN mkdir -p /app/logs \
 && chown -R appuser:appuser /app/logs

ENV HF_HOME=/home/appuser/.cache/huggingface
VOLUME $HF_HOME
RUN mkdir -p $HF_HOME \
 && chown -R appuser:appuser $HF_HOME

# Ensure our new site-packages are on PYTHONPATH
ENV PATH=/usr/local/bin:$PATH
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:/app
ENV LOG_FILE=/app/logs/mcp_server.log

# Switch to appuser
USER appuser

EXPOSE 9999
CMD ["python", "mcp_server.py"]
