# Docker base image
FROM python:3.9-slim-buster

# Working directory
WORKDIR /usr/src/app

# Installing python dependencies
COPY requirements.txt /usr/src/app/

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

# Copy source code 
COPY . /usr/src/app

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
