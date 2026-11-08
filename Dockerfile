FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu \
    && useradd --create-home worker
COPY baseline.py model.py synthesizer.py visual_features.py train.py ./
COPY backend/api.py backend/api.py
USER worker
EXPOSE 8000
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "8", "--timeout-keep-alive", "5"]
