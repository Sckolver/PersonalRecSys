FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts /opt/artifacts

ENV PYTHONUNBUFFERED=1
ENV ARTIFACTS_DIR=/opt/artifacts
CMD ["python", "-m", "src.app"]
