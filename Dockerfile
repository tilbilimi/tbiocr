FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
COPY packages.txt /app/packages.txt
RUN apt-get update \
    && xargs -a /app/packages.txt apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir python-doctr[torch]
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]