## Getting Started

#### Training && Inference on PASCAL VOC:

```shell
$ cd ./model/voc8.res50v3+.CPS
$ bash script.sh
```

- The tensorboard file is saved in `log/tb/` directory.
- In `script.sh`, you need to specify some variables, such as the path to your data dir, the path to your snapshot dir that stores checkpoints, etc.