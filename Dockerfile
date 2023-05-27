# create folder 
RUN cd smrfus
RUN mkdir model

ENV MODEL_DIR=/home/jovyan/my-model
ENV MODEL_FILE_LDA=model/ckpt

RUN pip install joblib vedo

# vedo: mesh

COPY inference.py ./inference.py

RUN python3 inference.py test_volume.mhd
