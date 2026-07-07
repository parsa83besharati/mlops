FROM python:3.11-slim

WORKDIR /app

# جلوگیری از ساخت pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# نصب ابزارهای لازم
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# نصب نسخه‌های سازگار
RUN pip install --no-cache-dir \
    mlflow==2.22.0 \
    scikit-learn==1.7.2 \
    pandas==2.3.3 \
    numpy==2.4.6 \
    fastapi==0.115.12 \
    starlette==0.46.2 \
    uvicorn==0.34.3

# کپی مدل
COPY deployment_model /app/model

EXPOSE 5001

CMD ["mlflow", "models", "serve","-m", "/app/model","--host", "0.0.0.0","--port", "5001","--env-manager", "local"]