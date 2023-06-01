# Ultrasound to point cloud (of pipes)

**[Site under construction; inquires welcomed]**

1. Clone
```
git clone https://github.com/lisatwyw/smrvis.git
cd smrvis
```

2. Build 
```
docker build --tag docker-test -f Dockerfile .
```
 
3. Test

```
docker run docker-test python3 source/test.py models/rand2.npz output_test_file .25 16
```

## Usage notes

Arguments to test.py:
1) ```models2/rand2.npz``` 
   - Filename of the volume to be processed; best if read with a meta header file (MHD) 
   - Sample of MHD file is provided [here](samples/scan_001.mhd) 
   - ```models/rand2.npz``` provided for test runs only   
2) ```output_test_file``` 
   - Prefix of output file that will save the coordinates of the extracted point cloud
3) ```0.25```: 
   - Threshold on the probablistic output $y$ (value may change depending on model used)
4) ```16```:
   - Model will cast predictions on 16 slices at a time (use higher/lower number if more/limited RAM is available)

The script currently outputs coordinates of the point cloud (challenge evaluates Chamfer distance between reference and extracted point cloud).

To visualize a mesh instead, the Veko package could be used to extract an isosurface out of the probablistic output $y$, as demonstrated in the code snippet in the [report](log/Report.pdf) 


***

Above was last tested in [Docker classroom](https://training.play-with-docker.com/beginner-linux/) on May 30, 2023. 

The following packages will be installed using ```req_slim.txt```

'absl-py==1.4.0', 'astunparse==1.6.3', 'cachetools==5.3.1','certifi==2023.5.7', 'charset-normalizer==3.1.0', 'contourpy==1.0.7', 'cycler==0.11.0', 'flatbuffers==23.5.26', 'fonttools==4.39.4', 'gast==0.4.0', 'google-auth-oauthlib==1.0.0', 'google-auth==2.19.0', 'google-pasta==0.2.0', 'grpcio==1.54.2', 'h5py==3.8.0', 'idna==3.4', 'importlib-metadata==6.6.0', 'importlib-resources==5.12.0', 'jax==0.4.10', 'keras==2.12.0', 'kiwisolver==1.4.4', 'libclang==16.0.0', 'markdown==3.4.3', 'markupsafe==2.1.2', 'matplotlib==3.7.1', 'ml-dtypes==0.1.0', 'numpy==1.23.5', 'oauthlib==3.2.2', 'opt-einsum==3.3.0', 'packaging==23.1', 'pillow==9.5.0', 'pip==22.0.4', 'platformdirs==3.5.1', 'pooch==1.7.0', 'protobuf==4.23.2', 'pyasn1-modules==0.3.0', 'pyasn1==0.5.0', 'pyparsing==3.0.9', 'python-dateutil==2.8.2', 'pyvista==0.39.1', 'requests-oauthlib==1.3.1', 'requests==2.31.0', 'rsa==4.9', 'scipy==1.10.1', 'scooby==0.7.2', 'setuptools==57.5.0', 'simpleitk==2.2.1', 'six==1.16.0', 'tensorboard-data-server==0.7.0', 'tensorboard==2.12.3', 'tensorflow-estimator==2.12.0', 'tensorflow-io-gcs-filesystem==0.32.0', 'tensorflow==2.12.0', 'termcolor==2.3.0', 'typing-extensions==4.6.2', 'urllib3==1.26.16', 'vtk==9.2.6','werkzeug==2.3.4', 'wheel==0.40.0', 'wrapt==1.14.1', 'zipp==3.15.0']

***

# References  

- ["A quick and easy build of a Docker container with a simple machine learning model" by Xavier Vasque](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)

- [Buiild and run a docker container, by Xavier Vasques](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)



***

Example outputs

![image width="200"](https://github.com/lisatwyw/test/assets/38703113/ba3e3b5e-f6e2-48d8-b2aa-76a5481a163c)

![image width="200"](https://github.com/lisatwyw/test/assets/38703113/d8a59e61-32c7-4ee7-aa99-ff2b38afb5c2)


