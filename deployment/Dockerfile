FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Run the Flask app
# CMD ["python", "serve_model.py"]
CMD ["python", "dashboard.py"]

# # Start FastAPI server
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# optimized
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t fraud-detection-api .
# docker run -p 5000:5000 fraud-detection-api
# docker build -t fraud-detection-dashboard .
# docker run -p 8050:8050 fraud-detection-dashboard