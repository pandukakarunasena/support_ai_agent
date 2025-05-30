# Stage 1: Build stage - with build tools
FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .

# Step 1: Update package lists
RUN apt-get update

# Step 2: Install build-essential for compiling packages
RUN apt-get install -y --no-install-recommends build-essential

# Step 3: Install Python dependencies
# The --user flag installs packages to /root/.local (as the builder runs as root),
# which aligns with your COPY --from=builder /root/.local command later.
# The --prefix=/install flag was likely conflicting or unnecessary with --user.
RUN pip install --user --no-cache-dir -r requirements.txt

# Step 4: Clean up apt packages and lists
RUN apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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