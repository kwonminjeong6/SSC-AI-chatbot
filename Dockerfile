FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /srv

# CPU 전용 torch를 먼저 설치합니다 (기본 torch보다 이미지가 수 GB 작음)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# 임베딩 모델을 이미지 안에 미리 내려받습니다.
# 이렇게 해야 서버가 새로 뜰 때(콜드스타트) 2GB를 다시 내려받지 않습니다.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY app ./app
COPY frontend ./frontend

# Cloud Run은 PORT 환경변수로 포트를 알려줍니다 (기본 8080)
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
