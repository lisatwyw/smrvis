FROM python:3.8-slim

RUN mkdir -p models
RUN mkdir -p source
RUN mkdir -p samples

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY ./req_slim.txt /req_slim.txt
COPY ./req.txt /req.txt
COPY ./source /source

RUN pip3 install --no-cache-dir -r /req_slim.txt
CMD ['python3', '/source/test.py']
EXPOSE 8080

# python3 infer.py test_volume.mhd
