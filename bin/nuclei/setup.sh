#!/usr/bin/env bash

# Install Nvidia

sudo apt-get install nvidia-352 nvidia-modprobe

# Install docker: Log out and log in after the below commands

sudo apt-get update

sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update

sudo apt-get install -y docker-ce

sudo docker run hello-world

sudo groupadd docker

sudo usermod -aG docker $USER


# Nvidia-docker

# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f

sudo apt-get purge -y nvidia-docker

# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
sudo apt-get install -y nvidia-docker2

sudo pkill -SIGHUP dockerd

# Test nvidia-smi with the latest official CUDA image
docker run --runtime=nvidia  nvidia/cuda nvidia-smi


# clone the NucleiDetectron project to host
git clone https://github.com/gangadhar-p/NucleiDetectron.git


# Download Data
# Note: maske sure you set up your kaggle account at sudo nano /home/ubuntu/.kaggle

pip install kaggle

cd NucleiDetectron/lib/datasets/data
kaggle datasets download -w -d gangadhar/nuclei-segmentation-in-microscope-cell-images
unzip Nuclei.zip

kaggle datasets download -w -d gangadhar/nuclei-detectron-models-for-2018-data-science-bowl
unzip models.zip

# cleanup
rm Nuclei.zip
rm nuclei-segmentation-in-microscope-cell-images.zip
rm nuclei-detectron-models-for-2018-data-science-bowl.zip
rm color_wikipedia.zip
rm models.zip

# Build a container
cd ../../../../
docker build -t nuclei-detectron:c2-cuda9-cudnn7 -f NucleiDetectron/docker/Dockerfile  .

# Test docker image with the following command
nvidia-docker run --rm -it nuclei-detectron:c2-cuda9-cudnn7 python2 tests/test_batch_permutation_op.py

nvidia-docker run -it nuclei-detectron:c2-cuda9-cudnn7 /bin/bash


# Run the below commands for training and testing inside the container
chmod +x bin/nuclei/train.sh && ./bin/nuclei/train.sh -e 1_aug_gray_0_5_0 -v 1_aug_gray_0_5_0 -g 1 &

chmod +x bin/nuclei/test.sh && ./bin/nuclei/test.sh -e 1_aug_gray_1_5_1_stage_2_v1 -v 1_aug_gray_1_5_1_stage_2_v1 -g 1 &

# look for logs as below
tail -f /detectron/lib/datasets/data/logs/test_log

# look for test results in the below folder
# /detectron/lib/datasets/data/results/1_aug_gray_1_5_1_stage_2_v1/test/nuclei_stage_2_test/generalized_rcnn/vis