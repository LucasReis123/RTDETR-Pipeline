# Ambiente de treinamento do RT-DETR.
#
# As wheels do PyTorch já trazem o runtime do CUDA/cuDNN, então a imagem base é um Python
# comum e a versão de CUDA é escolhida só pelo índice de onde o torch é baixado. O host
# precisa apenas do driver NVIDIA + NVIDIA Container Toolkit.
#
#   TORCH_CUDA   cu118 | cu121 | cu124 | cu126 | cu128 | cpu
#   TORCH_VERSION  vazio = mais recente disponível para o TORCH_CUDA escolhido

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG TORCH_CUDA=cu126
ARG TORCH_VERSION=
ARG TORCHVISION_VERSION=

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# Bibliotecas de sistema exigidas pelo OpenCV (dependência do Ultralytics)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
        "torch${TORCH_VERSION:+==${TORCH_VERSION}}" \
        "torchvision${TORCHVISION_VERSION:+==${TORCHVISION_VERSION}}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

WORKDIR /workspace/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./

CMD ["python", "main.py"]
