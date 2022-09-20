# Docker base image
FROM python:3.9

# Working directory
WORKDIR /usr/src/app

# Installing python dependencies
COPY requirements.txt /usr/src/app/

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# Copy source code 
COPY . /usr/src/app


EXPOSE 8000

ENTRYPOINT  ["gunicorn", "--bind=0.0.0.0:8000", "app:app"]
