FROM python:3.10-slim

COPY . .

RUN    apt-get -y update \
    && apt-get -y install git \
    && pip install .

ENTRYPOINT ["CMA"]
