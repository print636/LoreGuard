FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY alembic.ini .
COPY migrations migrations
COPY app app
COPY data data
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
