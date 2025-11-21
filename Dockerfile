# --- Build Stage ---
# This stage installs dependencies, including build-time requirements.
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /usr/src/app

# Install system dependencies required for PDF processing and other libraries.
# - libmagic-dev is for python-magic (used by unstructured)
# - poppler-utils is for PDF manipulation (used by unstructured)
# - build-essential is for compiling dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
# This stage creates the final, lean production image.
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install the same system dependencies required at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Make the virtual environment's binaries accessible
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser/app

# Copy the application code into the final image
COPY ./app ./app

# Expose the port the application runs on
EXPOSE 8000

# Define the command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
