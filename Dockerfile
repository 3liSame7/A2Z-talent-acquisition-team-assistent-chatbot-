# Base Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r app/requirements.txt

# Expose the required port (e.g., for Streamlit)
EXPOSE 8501

# Command to run your app
CMD ["streamlit", "run", "app/main.py"]
