FROM python:3.11-alpine

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apk update \
    && apk add --no-cache \
        build-base \
        libffi-dev \
        openssl-dev \
        bash \
    && rm -rf /var/cache/apk/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD [ "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000" ]
