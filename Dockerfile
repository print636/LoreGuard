FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN addgroup --system loreguard \
    && adduser --system --ingroup loreguard --home /app loreguard
COPY --chown=loreguard:loreguard alembic.ini .
COPY --chown=loreguard:loreguard migrations migrations
COPY --chown=loreguard:loreguard app app
COPY --chown=loreguard:loreguard data data
RUN mkdir -p /app/data/account-model-keys \
    && chown -R loreguard:loreguard /app/data/account-model-keys
USER loreguard
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
