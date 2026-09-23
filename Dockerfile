FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml constraints.txt ./
COPY src/ ./src/

RUN pip install --no-cache-dir -c constraints.txt . \
    && pip check

COPY models/medical_triage_model.joblib ./models/medical_triage_model.joblib

EXPOSE 8000

CMD ["uvicorn", "medical_triage.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]