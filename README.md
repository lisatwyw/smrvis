# test

[Last tested May 26, 2023]

Adopted from post written by [Xavier Vasques](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)

1. Clone
```
$ git clone https://github.com/lisatwyw/test.git
$ cd test
```

2. Build (includes pip installations)
```
docker build --tag docker-test -f Dockerfile .
```
 
3. Test on new data by calling ```inference.py```
```
docker run docker-test python3 inference.py
```


# References  

- ["A quick and easy build of a Docker container with a simple machine learning model" by Xavier Vasque](https://towardsdatascience.com/build-and-run-a-docker-container-for-your-machine-learning-model-60209c2d7a7f)
