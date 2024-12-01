#!/bin/bash

container_name="LASKV"
image_name_version="cuda11.8.0-devel-ubuntu22.04-lyj:0.2"
vscode_port=10002

# 暂停并删除容器
docker stop ${container_name} && docker rm ${container_name}

# 新建挂载目录
data_mount_path="/home/lyj/docker/${container_name}/data/"
data_mount_path_docker="/data/"
if [ ! -d ${data_mount_path} ]; then
    mkdir -p ${data_mount_path}
fi

project_mount_path="/home/lyj/project/"
project_mount_path_docker=${project_mount_path}
if [ ! -d ${project_mount_path} ];then
    mkdir -p ${project_mount_path}
fi

huggingface_mount_path="/home/lyj/.cache/huggingface/"
huggingface_mount_path_docker=${huggingface_mount_path}
if [ ! -d ${huggingface_mount_path} ];then
    mkdir -p ${huggingface_mount_path}
fi

# 新建并进入容器
docker run -itd \
    --name ${container_name} \
    --runtime=nvidia \
    --gpus all \
    -p ${vscode_port}:10008 \
    -v ${data_mount_path}:${data_mount_path_docker}:rw \
    -v ${project_mount_path}:${project_mount_path_docker}:rw \
    -v ${huggingface_mount_path}:${huggingface_mount_path_docker}:rw \
    ${image_name_version}
docker exec -it ${container_name} /bin/bash


docker exec -it LASKV /bin/bash

# docker run -itd \
#     --name cuda_test \
#     --runtime=nvidia \
#     --gpus all \
#     nvidia/cuda:11.8.0-devel-ubuntu22.04
# docker exec -it cuda_test /bin/bash