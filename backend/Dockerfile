FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:${PATH}"

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

RUN addgroup --system --gid 1001 appuser \
    && adduser --system --uid 1001 --ingroup appuser appuser

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
