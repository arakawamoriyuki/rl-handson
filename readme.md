
```
$ docker build -t rl-handson .
$ docker run -it -p 8888:8888 -v $(pwd):/app rl-handson /bin/bash
  $ xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --allow-root --ip=0.0.0.0 --port=8888
```
