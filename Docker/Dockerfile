FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
# FROM gitlab-registry.nrp-nautilus.io/lz1oceani/docker:cuda-gl-11.7.1-devel-ubuntu22.04

ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
RUN echo "Acquire::GzipIndexes \"false\"; Acquire::CompressionTypes::Order:: \"gz\";" > /etc/apt/apt.conf.d/docker-gzip-indexes

ENV TZ=America/Los_Angeles
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends -y \
    # Bare minimum Packages
    ca-certificates \
    git \
    git-lfs \
    emacs \
    meld \
    less \
    openssh-server \
    command-not-found \
    coreutils \
    pkg-config \
    xdg-user-dirs \
    ssh \
    sudo \
    wget \
    make \
    curl \
    libssl-dev \
    xterm \
    ffmpeg \
    nano \
    vim \
    unzip \
    zip \
    htop \
    tmux \
    bc \
    rsync \
    screen \
    pv \
    bash-completion \
    build-essential \
    software-properties-common \
    lsb-release \
    libtool \
    autoconf \
    libgl1-mesa-dev \
    freeglut3-dev \
    mesa-utils \
    vulkan-tools \
    libvulkan1 \
    libglvnd-dev \
    libvulkan-dev \
    libglfw3-dev \
    gcc \
    g++ \
    gcc-12 \
    g++-12 \
    gdb \
    cmake \
    net-tools \
    iputils-ping \
    zstd \
    aria2 \
    psmisc \
    lsof \
    graphviz-dev \
    libopenmpi-dev \
    swig \
    tig \
    espeak \
    tzdata \
    ttyd \
    s3fs \
    aria2 \
    xvfb \
    libaio-dev \
    apt-transport-https \
    gnupg \
    vlan \
    automake \
    dkms \
    xauth libibverbs* librdma* libibmad* libibumad* ibverbs-utils rdmacm-utils infiniband-diags perftest librdmacm-dev libibverbs-dev numactl libnuma-dev libnl-3-200 libnl-route-3-200 libnl-route-3-dev libnl-utils ibutils > /dev/null && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# https://github.com/haosulab/ManiSkill/issues/9
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN cd /root && curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-478.0.0-linux-x86_64.tar.gz && \
    tar -xf google-cloud-cli-478.0.0-linux-x86_64.tar.gz && rm google-cloud-cli-478.0.0-linux-x86_64.tar.gz && \
    ./google-cloud-sdk/install.sh -q

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "-c"]

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python=3.11 && \
    /opt/conda/bin/conda install mamba -n base -c conda-forge && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH

# Set up symlink to point to the correct python version
RUN ln -sf /opt/conda/bin/python /usr/bin/python
RUN ln -sf /opt/conda/bin/pip /usr/bin/pip
RUN ln -sf /opt/conda/bin/mamba /usr/bin/mamba
RUN ln -sf /opt/conda/bin/conda /usr/bin/conda
RUN ln -sf /opt/conda/bin/pytest /usr/bin/pytest
RUN ln -sf /opt/conda/bin/tensorboard /usr/bin/tensorboard

# Update git config
RUN git config --global http.sslverify false
RUN curl https://rclone.org/install.sh | bash

RUN mamba install -y -q mpi4py mpich scikit-image scikit-learn && mamba clean --all

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.3.0 torchvision torchaudio deepspeed==0.14.0 rclone-python

RUN cd /root && git clone https://github.com/lz1oceani/transformers.git && cd /root/transformers && \
    pip install -e . && pip cache purge

RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation
RUN MAX_JOBS=4 pip install --no-cache-dir -U xformers --index-url https://download.pytorch.org/whl/cu121

# Copy SSH private key into the Docker image
COPY id_rsa /root/.ssh/id_rsa

# Set permissions for the SSH private key
RUN chmod 600 /root/.ssh/id_rsa

# Configure SSH to ignore host key checking (optional but recommended)
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config

# Add GitHub's SSH key to known hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN cd /root/ && git clone -b depth git@github.com:lz1oceani/openvla.git
RUN cd /root/openvla && pip install -e .
RUN pip install scipy==1.10.1 && pip cache purge

RUN cd /root/ && git clone git@github.com:akshaygopalkr/SimplerEnv.git --recurse-submodules && \
    cd /root/SimplerEnv/ManiSkill2_real2sim && pip install -e . && \
    cd /root/SimplerEnv && pip install -e .
RUN cd /root/SimplerEnv && pip install -r requirements_full_install.txt

RUN cd /usr/local/bin/ && curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

RUN mkdir /root/.local && mkdir /root/.local/share
COPY lscolors.sh /root/.local/share/lscolors.sh

RUN echo "source ~/.local/share/lscolors.sh" >> ~/.bashrc
RUN echo 'alias ls="ls --color=auto"' >> ~/.bashrc
RUN echo "export MAX_JOBS=4" >> ~/.bashrc
RUN echo "export TZ=America/Los_Angeles" >> ~/.bashrc
RUN echo "export HF_DATASETS_CACHE=/ariesdv0/zhanling/hf-datasets" >> ~/.bashrc
RUN echo "export HF_HOME=/ariesdv0/zhanling/huggingface" >> ~/.bashrc

RUN echo "OMP_NUM_THREADS=64" >> ~/.bashrc
RUN echo "NUMEXPR_MAX_THREADS=64" >> ~/.bashrc
RUN echo "NCCL_NET=IB" >> ~/.bashrc
RUN echo "export PATH=$PATH:/root/google-cloud-sdk/bin" >> ~/.bashrc

RUN cd /root/openvla && rm -rf data && ln -s /ariesdv0/zhanling/oxe-data-converted ./data && ln -s /ariesdv0/zhanling/openvla-logs ./logs && ln -s /ariesdv0/zhanling/huggingface/token ./hf-token
RUN cd /root/SimplerEnv && ln -s /ariesdv0/zhanling/simpler_env_results ./results