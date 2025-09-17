# Deployment Guide

This guide provides instructions for deploying the House Price Prediction API to various platforms.

## Prerequisites

- Python 3.8+
- All dependencies from `requirements.txt`
- Trained models in the `models/` directory

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (if not already trained)
python train_models.py

# Start the API server
cd src
python -m api.main

# Or using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Production Deployment

### Using Gunicorn (Recommended)

```bash
# Install Gunicorn
pip install gunicorn

# Start with multiple workers
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

Build and run:

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

## Cloud Deployment Options

### AWS Lambda (Serverless)

1. Install Mangum for Lambda compatibility:
   ```bash
   pip install mangum
   ```

2. Modify `src/api/main.py` to add Lambda handler:
   ```python
   from mangum import Mangum
   
   # Add at the end of the file
   handler = Mangum(app)
   ```

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/[PROJECT-ID]/house-price-api
gcloud run deploy --image gcr.io/[PROJECT-ID]/house-price-api --platform managed
```

### Azure Container Instances

```bash
# Deploy to Azure
az container create --resource-group myResourceGroup --name house-price-api --image myregistry.azurecr.io/house-price-api:latest --cpu 1 --memory 1 --ports 8000
```

### Heroku

1. Create `Procfile`:
   ```
   web: gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
   ```

2. Deploy:
   ```bash
   heroku create house-price-api
   git push heroku main
   ```

## Environment Variables

Set these environment variables for production:

```bash
# Optional: API configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_PATH=/path/to/models

# Optional: Logging
export LOG_LEVEL=INFO
```

## Health Monitoring

The API provides health check endpoints:

- `GET /health` - Basic health check
- `GET /` - Application status
- `GET /model/info` - Model information

## Performance Optimization

1. **Model Loading**: Models are loaded once at startup
2. **Caching**: Consider adding Redis for prediction caching
3. **Scaling**: Use multiple workers with Gunicorn
4. **Load Balancing**: Use nginx or cloud load balancers

## Security Considerations

1. **Input Validation**: All inputs are validated using Pydantic
2. **Error Handling**: Graceful error responses without exposing internals
3. **CORS**: Configure CORS for web frontend integration
4. **Rate Limiting**: Consider adding rate limiting for production use

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure models are in the correct directory
2. **Memory issues**: Reduce number of workers or optimize model size
3. **Port conflicts**: Change the port in deployment configuration

### Logs

Check application logs for debugging:

```bash
# Local development
tail -f app.log

# Docker
docker logs <container-id>

# Cloud platforms
# Check platform-specific logging solutions
```
