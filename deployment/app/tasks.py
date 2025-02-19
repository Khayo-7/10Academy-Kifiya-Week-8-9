from celery import Celery

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task
def process_fraud_detection(data):
    """Asynchronous fraud detection task."""
    # Run fraud detection here...
    return {"fraud": False, "confidence": 0.92}

# celery -A api.tasks worker --loglevel=info
