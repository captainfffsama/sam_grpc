FROM  pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime as rpc
RUN sed -i "s@http://.*archive.ubuntu.com@http://mirrors.bfsu.edu.cn@g" /etc/apt/sources.list \
    && sed -i "s@http://.*security.ubuntu.com@http://mirrors.bfsu.edu.cn@g" /etc/apt/sources.list \
    # && rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    # && apt-key del 7fa2af80 \
    # && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
    # && dpkg -i cuda-keyring_1.0-1_all.deb \
    # && rm cuda-keyring_1.0-1_all.deb \
    && apt update && apt install -y \
    git \
    wget \
    monit \
    # libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev\
    # libglib2.0-0 \
    # libsm6 \
    # libxext6 \
    # libxrender-dev \
    # libgl1 \
    # ffmpeg \
    # build-essential \
    # ninja-build \
    # libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && cd /root \
    && git config --global http.sslverify false \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir grpcio==1.37.0 \
    && pip install --no-cache-dir grpcio-tools==1.37.0 \
    && pip install --no-cache-dir timm==0.9.2 \
    && pip install --no-cache-dir opencv-python-headless \
    && pip install git+https://github.com/ChaoningZhang/MobileSAM.git\
    && pip install --no-cache-dir pid  \
    && pip install git+https://github.com/captainfffsama/sam_grpc.git \
    && conda clean --all

RUN mkdir /root/mount \
    && mkdir /root/mount/sam_weight \
    && cd /root/mount/sam_weight \
    # && wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    && wget https://soft.chiebot.com:10000/model_weight/mobileSAM/mobile_sam.pt



FROM rpc as final
COPY ./utils/monitrc /root/monitrc
COPY ./utils/server_manager /root/server_manager
COPY ./utils/cfg.yaml /root/cfg.yaml
RUN cd /root \
    && chmod -R a+x ./server_manager/* \
    && mv /root/monitrc /etc/monit/monitrc \
    && chmod 0700 /etc/monit/monitrc
#修改时区
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
EXPOSE 52018
#monit端口
EXPOSE 2814
ENTRYPOINT [ "sh","/root/server_manager/system_entry_point.sh"]
