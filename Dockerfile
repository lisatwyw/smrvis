# create folder 
RUN cd smrfus
RUN mkdir model

# declare environment variables, if needed
# ENV MODEL_FILE_LDA=model/ckpt

RUN pip install tensorflow pyvista vedo
# vedo: mesh

COPY infer.py ./infer.py

RUN python3 infer.py test_volume.mhd
