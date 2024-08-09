FROM lingzhan/openvla
ENV NVIDIA_DRIVER_CAPABILITIES=all
RUN apt-get install --no-install-recommends -y libvulkan1 libglvnd-dev

# Vulkan issues
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Copy SSH private key into the Docker image
COPY id_rsa /root/.ssh/id_rsa

# Set permissions for the SSH private key
RUN chmod 600 /root/.ssh/id_rsa

# Configure SSH to ignore host key checking (optional but recommended)
RUN echo "Host github.com\n\tStrictHostKeyChecking no\n" >> /root/.ssh/config

# Add GitHub's SSH key to known hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN rm -rf /root/SimplerEnv
RUN cd /root/openvla && git remote set-url origin git@github.com:lz1oceani/openvla.git && git checkout -b depth  && \
    git pull origin depth
RUN cd /root/ && git clone git@github.com:akshaygopalkr/SimplerEnv.git --recurse-submodules
RUN pip uninstall -y transformers && pip install transformers==4.40.1
RUN pip uninstall -y flash-attn
RUN pip install flash-attn -U
RUN pip install mediapy==1.2.0
RUN pip install git+https://github.com/nathanrooy/simulated-annealing
RUN cd /root/SimplerEnv && ln -s /ariesdv0/zhanling/simpler_env_results ./results