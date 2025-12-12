# Middleware Module

## Purpose
Custom middleware directory for request processing, authentication, rate limiting, and other cross-cutting concerns (currently empty - using FastAPI built-in middleware).

## Key Files

**No custom middleware files currently.** Felix API uses FastAPI's built-in middleware capabilities configured in [main.py](../main.py):

### Current Middleware (via FastAPI)

**CORS Middleware** - Configured in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Authentication** - Handled via dependencies in [dependencies.py](../dependencies.py):
- `verify_api_key()`: API key validation using FastAPI's dependency injection
- `HTTPBearer` security scheme from `fastapi.security`

## Key Concepts

### Why No Custom Middleware?
Felix's current needs are well-served by:
1. **FastAPI dependencies**: Fine-grained per-route control for auth
2. **Built-in middleware**: CORS, trusted host, gzip compression
3. **Simplicity**: Fewer layers means easier debugging

### When to Add Custom Middleware
Consider adding custom middleware for:
- **Rate limiting**: Prevent API abuse (e.g., max 100 requests/minute per API key)
- **Request logging**: Detailed audit trail of all API calls
- **Response compression**: Automatic gzip for large responses
- **Request ID tracking**: Unique ID per request for distributed tracing
- **Metrics collection**: Prometheus-style metrics (request count, latency, errors)
- **Custom headers**: Add standard headers to all responses

### Example Custom Middleware
```python
# Example: Request logging middleware
from fastapi import Request
import time
import logging

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {process_time:.3f}s "
        f"with status {response.status_code}"
    )

    return response
```

## Related Modules
- [main.py](../main.py) - FastAPI application with middleware configuration
- [dependencies.py](../dependencies.py) - Authentication and dependency injection
- [routers/](../routers/) - Endpoint handlers that benefit from middleware
