FROM mendelxu/pytorch:d2_nvcr_2008

RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install cython scipy shapely timm h5py submitit scikit-image wandb setuptools numpy Pillow pycocotools~=2.0.4 fvcore tabulate tqdm ftfy regex open_clip_torch cityscapesscripts tensorboard gradio fastapi uvicorn

RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME
RUN git clone https://github.com/deepanaishtaweera/SAN.git app

WORKDIR $HOME/app

RUN git checkout deepana/fixes

EXPOSE 7860


# ENV GRADIO_SERVER_NAME=0.0.0.0
# RUN echo "gradio app.py">>run.sh
# CMD ["script","-c","sh run.sh","/dev/null"]

CMD ["python", "api.py"]
