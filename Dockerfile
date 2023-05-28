FROM scratch

RUN mkdir models
RUN mkdir source

RUN pip install tensorflow pyvista vedo

COPY infer.py ./infer.py

RUN python3 infer.py test_volume.mhd
