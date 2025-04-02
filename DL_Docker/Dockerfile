FROM ubuntu:latest

WORKDIR /app

RUN apt-get update -y
RUN apt-get install -y python3 python3-venv python3-pip python3-dev build-essential hdf5-tools libgl1 libgtk2.0-dev
COPY . /app
RUN python3 -m venv /app/venv \
    && /app/venv/bin/pip install --upgrade pip \
    && /app/venv/bin/pip install -r requirements.txt


EXPOSE 8080
CMD ["/app/venv/bin/python", "my_api.py"]
