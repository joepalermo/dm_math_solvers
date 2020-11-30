Seeded with code from a prior project by Helen Ngo, Joseph Palermo and Michael Jia, with support from Rayhane Mama (https://github.com/mathemakitten/transformers-mathematics).

Download data
wget https://storage.cloud.google.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz

Docker commands
nvidia-docker run -it -p 6006:6006 -v /home/jpalermo:/home/ tensorflow/tensorflow:latest-gpu /bin/bash
nvidia-docker run -it -p 6006:6006 -v /home/jpalermo:/home/ math /bin/bash
docker run -gpus all -it -p 6006:6006 -v /home/jpalermo:/home/ math /bin/bash

tensorboard --logdir=/home/dm_math_solvers/experiment_results --bind_all
tensorboard --logdir=logs/gradient_tape --bind_all