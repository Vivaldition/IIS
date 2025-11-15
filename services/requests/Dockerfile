FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install requests

COPY services/requests/requests.py .

CMD ["python", "requests.py"]
