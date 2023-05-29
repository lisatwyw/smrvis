FROM scratch

RUN mkdir -p models
RUN mkdir -p source

RUN pip install tensorflow pyvista vedo

COPY source/infer.py source/infer.py

RUN python3 infer.py test_volume.mhd
