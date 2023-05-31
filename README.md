# Ultrasound to point cloud (of pipes)

**[Site under construction]**


1. Clone
```
git clone https://github.com/lisatwyw/smrvis.git
cd smrvis
```

2. Build (includes pip installations)
```
docker build --tag docker-test -f Dockerfile .
```
 
3. Test

```
docker run docker-test python3 source/test.py
```

4. [Pending] Apply on data by calling ```infer.py```
```
docker run docker-test python3 source/infer.py
```

***

Above was last tested in [Docker classroom](https://training.play-with-docker.com/beginner-linux/). 

The following packages will be installed using req_slim.txt

'absl-py==1.4.0', 'astunparse==1.6.3', 'cachetools==5.3.1','certifi==2023.5.7', 'charset-normalizer==3.1.0', 'contourpy==1.0.7', 'cycler==0.11.0', 'flatbuffers==23.5.26', 'fonttools==4.39.4', 'gast==0.4.0', 'google-auth-oauthlib==1.0.0', 'google-auth==2.19.0', 'google-pasta==0.2.0', 'grpcio==1.54.2', 'h5py==3.8.0', 'idna==3.4', 'importlib-metadata==6.6.0', 'importlib-resources==5.12.0', 'jax==0.4.10', 'keras==2.12.0', 'kiwisolver==1.4.4', 'libclang==16.0.0', 'markdown==3.4.3', 'markupsafe==2.1.2', 'matplotlib==3.7.1', 'ml-dtypes==0.1.0', 'numpy==1.23.5', 'oauthlib==3.2.2', 'opt-einsum==3.3.0', 'packaging==23.1', 'pillow==9.5.0', 'pip==22.0.4', 'platformdirs==3.5.1', 'plyfile==0.9', 'pooch==1.7.0', 'protobuf==4.23.2', 'pyasn1-modules==0.3.0', 'pyasn1==0.5.0', 'pyparsing==3.0.9', 'python-dateutil==2.8.2', 'pyvista==0.39.1', 'requests-oauthlib==1.3.1', 'requests==2.31.0', 'rsa==4.9', 'scipy==1.10.1', 'scooby==0.7.2', 'setuptools==57.5.0', 'simpleitk==2.2.1', 'six==1.16.0', 'tensorboard-data-server==0.7.0', 'tensorboard==2.12.3', 'tensorflow-estimator==2.12.0', 'tensorflow-io-gcs-filesystem==0.32.0', 'tensorflow==2.12.0', 'termcolor==2.3.0', 'typing-extensions==4.6.2', 'urllib3==1.26.16', 'vtk==9.2.6','werkzeug==2.3.4', 'wheel==0.40.0', 'wrapt==1.14.1', 'zipp==3.15.0']

***


# References  

- ["A quick and easy build of a Docker container with a simple machine learning model" by Xavier Vasque](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)

- [Buiild and run a docker container, by Xavier Vasques](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)




