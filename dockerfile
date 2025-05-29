# Stage 1: Build stage - with build tools
FROM python:3.10-slim AS builder

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# (Removed pre-download of SentenceTransformer to keep build lean)

# ---
# Stage 2: Final stage - runtime image
FROM python:3.10-slim

WORKDIR /app

# Create a non-root user and group
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser

# Copy installed Python packages from the builder stage's user site-packages
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code including your main script
COPY . .

# Set up Hugging Face cache directory as a mountable volume
ENV HF_HOME=/home/appuser/.cache/huggingface
VOLUME /home/appuser/.cache/huggingface

RUN mkdir -p /home/appuser/.cache/huggingface \
 && chown -R appuser:appuser /home/appuser/.cache/huggingface

# Create logs directory and set ownership
RUN mkdir -p logs && chown -R appuser:appuser logs /app
ENV LOG_FILE=/app/logs/mcp_server.log

# Set Python path to include user's local site-packages and application
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/home/appuser/.local/lib/python3.10/site-packages:/app

# Switch to non-root user
USER appuser

# Expose the port the app runs on (default 9999)
EXPOSE 9999
ENV MCP_HOST="0.0.0.0"
ENV MCP_PORT="9999"

# Command to run the application
CMD ["python", "mcp_server.py"]
