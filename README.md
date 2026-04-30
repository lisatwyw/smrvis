# Ultrasound to point cloud (of pipes)

**[Site under construction; inquiries welcomed]**

1. Clone
```
git clone https://github.com/lisatwyw/smrvis.git
cd smrvis
```

2. Build 
```
docker build --tag docker-test -f Dockerfile .
```

 <details>
 <summary>Outputs</summary>

 ```
[+] Building 295.8s (19/19) FINISHED                                                                                     docker:default
 => [internal] load build definition from Dockerfile                                                                               0.0s
 => => transferring dockerfile: 526B                                                                                               0.0s
 => [internal] load metadata for docker.io/library/python:3.8-slim                                                                 1.1s
 => [auth] library/python:pull token for registry-1.docker.io                                                                      0.0s
 => [internal] load .dockerignore                                                                                                  0.0s
 => => transferring context: 2B                                                                                                    0.0s
 => [ 1/13] FROM docker.io/library/python:3.8-slim@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f         3.5s
 => => resolve docker.io/library/python:3.8-slim@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f           0.0s
 => => sha256:3971691a363796c39467aae4cdce6ef773273fe6bfc67154d01e1b589befb912 248B / 248B                                         0.1s
 => => sha256:a3f1dfe736c5f959143f23d75ab522a60be2da902efac236f4fb2a153cc14a5d 14.53MB / 14.53MB                                   0.9s
 => => sha256:030d7bdc20a63e3d22192b292d006a69fa3333949f536d62865d1bd0506685cc 3.51MB / 3.51MB                                     0.5s
 => => sha256:302e3ee498053a7b5332ac79e8efebec16e900289fc1ecd1c754ce8fa047fcab 29.13MB / 29.13MB                                   1.3s
 => => extracting sha256:302e3ee498053a7b5332ac79e8efebec16e900289fc1ecd1c754ce8fa047fcab                                          1.1s
 => => extracting sha256:030d7bdc20a63e3d22192b292d006a69fa3333949f536d62865d1bd0506685cc                                          0.1s
 => => extracting sha256:a3f1dfe736c5f959143f23d75ab522a60be2da902efac236f4fb2a153cc14a5d                                          0.8s
 => => extracting sha256:3971691a363796c39467aae4cdce6ef773273fe6bfc67154d01e1b589befb912                                          0.0s
 => [internal] load build context                                                                                                  0.6s
 => => transferring context: 34.98MB                                                                                               0.5s
 => [ 2/13] RUN mkdir -p models                                                                                                    2.5s
 => [ 3/13] RUN mkdir -p source                                                                                                    0.2s
 => [ 4/13] RUN mkdir -p samples                                                                                                   0.3s
 => [ 5/13] RUN mkdir -p utils                                                                                                     0.4s
 => [ 6/13] RUN apt update &&     apt install --no-install-recommends -y build-essential gcc &&     apt clean && rm -rf /var/lib  21.9s
 => [ 7/13] COPY ./req_slim.txt /req_slim.txt                                                                                      0.1s
 => [ 8/13] COPY ./req.txt /req.txt                                                                                                0.0s
 => [ 9/13] COPY ./source /source                                                                                                  0.0s
 => [10/13] COPY ./models /models                                                                                                  0.5s
 => [11/13] COPY ./samples /samples                                                                                                0.0s
 => [12/13] COPY ./utils /utils                                                                                                    0.0s
 => [13/13] RUN pip3 install --no-cache-dir -r /req_slim.txt                                                                      80.3s
 => exporting to image                                                                                                           183.6s
 => => exporting layers                                                                                                          129.8s
 => => exporting manifest sha256:6af007097f34b44fe6e5a2a02295b4f293318afdd608d1e8d80de167b1ff81fd                                  0.0s
 => => exporting config sha256:7e3392255e65994c5d26967373d3423601eb1cfa27d3cd90f672989b68a70685                                    0.0s
 => => exporting attestation manifest sha256:1083295bace2f8dd6d6007645c301e7d1de85b56323bba727bb1c1ca0fd65f5d                      0.0s
 => => exporting manifest list sha256:3e760af3a56814062a72885459ddf6c874f05c7308f5d76bd74d71d6131c8b88                             0.0s
 => => naming to docker.io/library/docker-test:latest                                                                              0.0s
 => => unpacking to docker.io/library/docker-test:latest                                                                          53.5s
```


</details>

3. Test

```
docker run docker-test python3 source/test.py models/rand2.npz output_test_file .25 16
```

 <details>
 <summary>Outputs</summary>
```
2026-04-30 00:38:05.689320: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2026-04-30 00:38:06.028421: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2026-04-30 00:38:06.030009: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-04-30 00:38:09.828025: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT

Linux-6.8.0-1044-azure-x86_64-with-glibc2.34
Python 3.8.20 (default, Sep 27 2024, 06:05:23) 
[GCC 12.2.0]
Tensorflow 2.12.0
tf.Tensor([0.04552065 0.26180163 0.08348185], shape=(3,), dtype=float32)
Above equal to [0.04552065, 0.26180163, 0.08348185]
?

tf.Tensor(
[ 0.1260792  -0.0652371  -0.0849674   0.11226883  0.10652731  0.12007868
  0.1658944  -0.04231708], shape=(8,), dtype=float32)
Above equal to [ 0.1260792 , -0.0652371 , -0.0849674 ,  0.11226883,  0.10652731, 0.12007868,  0.1658944 , -0.04231708]
?
No filename provided; test data will be used...

Run in demo mode! 

Example usage: test.py input.mhd output_prefix 0.25 8
Predict in batches of 8 slices and apply global threshold of 0.25 on prb mask.
Reading ../models/rand2.npz
Will write to ../models/detected_pointcloud with global thres=0.2500
Input intensity range: 0.0 0.9999999996078431 (1280, 256, 256)
0 7 |8 15 |16 23 |24 31 |32 39 |40 47 |48 55 |56 63 |64 71 |72 79 |80 87 |88 95 |96 103 |104 111 |112 119 |120 127 |128 135 |136 143 |144 151 |152 159 |160 167 |168 175 |176 183 |184 191 |192 199 |200 207 |208 215 |216 223 |224 231 |232 239 |240 247 |248 255 |256 263 |264 271 |272 279 |280 287 |288 295 |296 303 |304 311 |312 319 |320 327 |328 335 |336 343 |344 351 |352 359 |360 367 |368 375 |376 383 |384 391 |392 399 |400 407 |408 415 |416 423 |424 431 |432 439 |440 447 |448 455 |456 463 |464 471 |472 479 |480 487 |488 495 |496 503 |504 511 |512 519 |520 527 |528 535 |536 543 |544 551 |552 559 |560 567 |568 575 |576 583 |584 591 |592 599 |600 607 |608 615 |616 623 |624 631 |632 639 |640 647 |648 655 |656 663 |664 671 |672 679 |680 687 |688 695 |696 703 |704 711 |712 719 |720 727 |728 735 |736 743 |744 751 |752 759 |760 767 |768 775 |776 783 |784 791 |792 799 |800 807 |808 815 |816 823 |824 831 |832 839 |840 847 |848 855 |856 863 |864 871 |872 879 |880 887 |888 895 |896 903 |904 911 |912 919 |920 927 |928 935 |936 943 |944 951 |952 959 |960 967 |968 975 |976 983 |984 991 |992 999 |1000 1007 |1008 1015 |1016 1023 |1024 1031 |1032 1039 |1040 1047 |1048 1055 |1056 1063 |1064 1071 |1072 1079 |1080 1087 |1088 1095 |1096 1103 |1104 1111 |1112 1119 |1120 1127 |1128 1135 |1136 1143 |1144 1151 |1152 1159 |1160 1167 |1168 1175 |1176 1183 |1184 1191 |1192 1199 |1200 1207 |1208 1215 |1216 1223 |1224 1231 |1232 1239 |1240 1247 |1248 1255 |1256 1263 |1264 1271 |1272 1279 |Model predictions range: 0.0 0.47297802567481995
210737 points will be saved to output_file ../models/detected_pointcloud
241,061 points saved to output_file test??

...
```

</details>


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

<details>
<summary>Packages when req_slim.txt is used</summary>
 The following packages will be installed using ```req_slim.txt```

'absl-py==1.4.0', 'astunparse==1.6.3', 'cachetools==5.3.1','certifi==2023.5.7', 'charset-normalizer==3.1.0', 'contourpy==1.0.7', 'cycler==0.11.0', 'flatbuffers==23.5.26', 'fonttools==4.39.4', 'gast==0.4.0', 'google-auth-oauthlib==1.0.0', 'google-auth==2.19.0', 'google-pasta==0.2.0', 'grpcio==1.54.2', 'h5py==3.8.0', 'idna==3.4', 'importlib-metadata==6.6.0', 'importlib-resources==5.12.0', 'jax==0.4.10', 'keras==2.12.0', 'kiwisolver==1.4.4', 'libclang==16.0.0', 'markdown==3.4.3', 'markupsafe==2.1.2', 'matplotlib==3.7.1', 'ml-dtypes==0.1.0', 'numpy==1.23.5', 'oauthlib==3.2.2', 'opt-einsum==3.3.0', 'packaging==23.1', 'pillow==9.5.0', 'pip==22.0.4', 'platformdirs==3.5.1', 'pooch==1.7.0', 'protobuf==4.23.2', 'pyasn1-modules==0.3.0', 'pyasn1==0.5.0', 'pyparsing==3.0.9', 'python-dateutil==2.8.2', 'pyvista==0.39.1', 'requests-oauthlib==1.3.1', 'requests==2.31.0', 'rsa==4.9', 'scipy==1.10.1', 'scooby==0.7.2', 'setuptools==57.5.0', 'simpleitk==2.2.1', 'six==1.16.0', 'tensorboard-data-server==0.7.0', 'tensorboard==2.12.3', 'tensorflow-estimator==2.12.0', 'tensorflow-io-gcs-filesystem==0.32.0', 'tensorflow==2.12.0', 'termcolor==2.3.0', 'typing-extensions==4.6.2', 'urllib3==1.26.16', 'vtk==9.2.6','werkzeug==2.3.4', 'wheel==0.40.0', 'wrapt==1.14.1', 'zipp==3.15.0']
</details>
 



***

[# Example outputs](log)

***

# References  

- ["A quick and easy build of a Docker container with a simple machine learning model" by Xavier Vasque](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)

- [Buiild and run a docker container, by Xavier Vasques](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)



# Citation

Please consider citing this work:

```
@article{tang2023smrvis,
  title={SMRVIS: Point cloud extraction from 3-D ultrasound for non-destructive testing},
  author={Tang, Lisa YW},
  journal={arXiv preprint arXiv:2306.04668},
  year={2023}
}
```

