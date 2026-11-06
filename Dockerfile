FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.10.0

COPY app/requirements.txt /app/app/requirements.txt

RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    fastapi==0.115.6 \
    "uvicorn[standard]==0.34.0" \
    pydantic==2.10.6 \
    transformers==5.5.4 \
    boto3==1.36.26

COPY app /app/app

ENV MODEL_URI=s3://model-storage/pythia-70m-command-parser
ENV MODEL_VERSION=v1
ENV S3_ENDPOINT=http://seaweedfs-s3:8333
ENV MAX_NEW_TOKENS=128

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]