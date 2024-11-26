#!/bin/bash

# 创建基础环境容器：cuda11.8.0-devel-ubuntu22.04-lyj
docker run -itd \
    --name cuda11.8.0-devel-ubuntu22.04-lyj \
    --runtime=nvidia \
    --gpus all \
    nvidia/cuda:11.8.0-devel-ubuntu22.04

docker exec -it cuda11.8.0-devel-ubuntu22.04-lyj /bin/bash

###########################################

# root设置密码
passwd
sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
source /root/.bashrc

# 添加普通用户
adduser --uid 1001 lyj
usermod -aG sudo lyj
usermod -aG root lyj
sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /home/lyj/.bashrc

# 安装必要软件
apt-get update
apt-get install -y openssh-server openssh-client ssh
apt-get install -y vim git wget sudo
rm -rf /var/lib/apt/lists/*

# 配置SSH服务
echo "Port 10008" >> /etc/ssh/sshd_config
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
/etc/init.d/ssh restart
echo "service ssh start" >> /root/.bashrc

# 给用户lyj安装miniconda3
su lyj
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh  # 选yes
source ~/.bashrc
rm Miniconda3-latest-Linux-x86_64.sh

# 设置git ssh公钥
ssh-keygen -t rsa -C '1025052251@qq.com'
cat  ~/.ssh/id_rsa.pub
# 打开github粘贴

#############################################

# 制作镜像：cuda11.8.0-devel-ubuntu22.04-lyj:0.1
exit
exit
docker stop cuda11.8.0-devel-ubuntu22.04-lyj
docker commit cuda11.8.0-devel-ubuntu22.04-lyj cuda11.8.0-devel-ubuntu22.04-lyj:0.1

#############################################

# 新建容器：based_evaluate_vscode
docker run -itd \
    --name based_evaluate_vscode \
    --runtime=nvidia \
    --gpus all \
    -p 10001:10008 \
    -v /home/lyj/docker/based_evaluate_vscode/data/:/data/:rw \
    -v /home/lyj/.cache/huggingface/:/home/lyj/.cache/huggingface/:rw \
    cuda11.8.0-devel-ubuntu22.04-lyj:0.1
docker exec -it based_evaluate_vscode /bin/bash
chown -R lyj:lyj /home/lyj/.cache/

su lyj
# 创建并激活conda虚拟环境
conda create -y -n based python=3.8.18
conda activate based

# 下载based源码与安装pytorch
cd ~ && mkdir project && cd project
git clone git@github.com:HazyResearch/based.git
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 安装flash_attn包和causal_conv1d包
# 宿主机上
cp flash_attn-2.5.2+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl ~/docker/based_evaluate_vscode/data/
cp causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl ~/docker/based_evaluate_vscode/data/
# 容器内
pip install /data/flash_attn-2.5.2+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install /data/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

# 安装based
cd ~/project/based 
pip install -e .
# 解决based源码的bug
pip install hydra-core
pip install --upgrade huggingface_hub
pip install -U huggingface_hub hf_transfer -i https://mirrors.aliyun.com/pypi/simple/

# 安装flash-attention扩展包
cd ~/project
git clone git@github.com:Dao-AILab/flash-attention.git
cd ~/project/flash-attention/csrc/layer_norm && pip install . && \
cd ~/project/flash-attention/csrc/fused_dense_lib && pip install .

# 安装evaluate
cd ~/project/based/evaluate
git submodule init
git submodule update
pip install -e .

# 生成evaluate的launch.py适配脚本
touch launch.sh
echo '#!/bin/bash' >> launch.sh
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> launch.sh
echo 'python launch.py --task swde --model "hazyresearch/based-360m"' >> launch.sh

# 复制测试脚本
# 宿主机上
cp ~/docker/based_py3.8.18_torch2.1.2_cu11.8_evaluate/build_files/based_test.py ~/docker/based_evaluate_vscode/data/
cp ~/docker/based_py3.8.18_torch2.1.2_cu11.8_evaluate/build_files/based_test.sh ~/docker/based_evaluate_vscode/data/
# 容器内
cp /data/based_test.py ~/project/based/notebooks/
cp /data/based_test.sh ~/project/based/notebooks/

# 测试base模型
cd ~/project/based/notebooks
sh -x based_test.sh

# 测试based模型的evaluate
cd ~/project/based/evaluate
sh -x launch.sh
