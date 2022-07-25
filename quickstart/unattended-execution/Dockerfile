FROM nvcr.io/nvidia/tensorflow:21.05-tf1-py3

# Install Keras
RUN pip install Keras==2.2.4


RUN mkdir /workload
COPY ./main.py /workload/

WORKDIR /workload
