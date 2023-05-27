# create folder 
RUN cd smrfus
RUN mkdir model

ENV MODEL_DIR=/home/jovyan/my-model
ENV MODEL_FILE_LDA=model/ckpt

RUN pip install joblib vedo tensorflow

# vedo: mesh

COPY infer.py ./infer.py

RUN python3 infer.py test_volume.mhd
