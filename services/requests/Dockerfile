FROM python:3.11-slim

WORKDIR /app
RUN pip install requests

COPY requests.py .

CMD ["python", "requests.py"]