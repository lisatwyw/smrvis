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
 
3. Test on new data by calling ```infer.py```
```
docker run docker-test python3 source/infer.py
```


# References  

- ["A quick and easy build of a Docker container with a simple machine learning model" by Xavier Vasque](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)

- [Buiild and run a docker container, by Xavier Vasques](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)

