import redis
import json
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def cache_result(key: str, data: dict, expire: int = 600):
    """Cache API response in Redis."""
    redis_client.setex(key, expire, json.dumps(data))

def get_cached_result(key: str):
    """Retrieve cached response from Redis."""
    cached_data = redis_client.get(key)
    return json.loads(cached_data) if cached_data else None
