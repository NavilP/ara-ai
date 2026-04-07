FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY my_env /app/my_env

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e /app/my_env

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"

CMD ["python", "-m", "my_env.server.app"]
