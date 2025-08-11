# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy API files and requirements
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/api.py .
COPY model.pkl .

# Expose port 5000
EXPOSE 5000

# Command to run the API with Gunicorn (production-grade server)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api:app"]

