
logger = logging.getLogger(__name__)

def log_request(data, response):
    """Log API requests and responses."""
    logger.info(f"Request: {data} | Response: {response}")