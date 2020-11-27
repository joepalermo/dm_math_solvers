FROM tensorflow/tensorflow:latest-gpu
RUN apt-get update
RUN apt-get install nano less
RUN pip install tqdm
