uv venv --python 3.8
source .venv/bin/activate
uv pip install -r requirements-torch.txt
uv pip install -r requirements.txt
uv pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# attach folder as docker 
docker run -it --gpus all --shm-size 8G -v "$(pwd)/SAN:/app" mendelxu/pytorch:d2_nvcr_2008 /bin/bash

docker build -f docker/app.Dockerfile -t san_app .
docker run -it --shm-size 4G -p 7860:7860  san_app 