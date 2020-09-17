FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update
RUN apt-get install -y git

RUN apt update
RUN apt -y upgrade
RUN apt install curl 

#RUN git clone https://github.com/my89/imSitu

WORKDIR imSitu

RUN mkdir baseline_models

#RUN curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_resnet_101 > baseline_models/baseline_resnet_101
#RUN curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_resnet_101 > data/models/imsitu/baseline_resnet_101
# RUN ./install.sh

#RUN conda create -n pytorch_env -c pytorch pytorch torchvision
#RUN source activate pytorch_env

RUN apt-get update
RUN apt-get install -y python3-dev python3-pip python3-nose python3-numpy python3-scipy
RUN pip3 install --upgrade pip pyyaml
RUN pip3 install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_HOME data/models/imsitu

ENTRYPOINT ["python3", "src/imsitu/imsitu_main.py"]
