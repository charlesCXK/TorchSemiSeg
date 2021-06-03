## Getting Started

### Training && Inference on PASCAL VOC:

```shell
$ cd ./model/voc8.res50v3+.CPS
$ bash script.sh
```

- The tensorboard file is saved in `log/tb/` directory.
- In `script.sh`, you need to specify some variables, such as the path to your data dir, the path to your snapshot dir that stores checkpoints, etc.

### Different partitions
To try other data partitions beside 1/8, you just need to change two variables in `config.py`:
```python
C.labeled_ratio = 8
C.nepochs = 34
```
Please note that, for fair comparison, we control the total iterations during training in each experiment similar (almost the same), including the supervised baseline and semi-supervised methods. Therefore, the nepochs for different partitions are different. We list the nepochs for different datasets and partitions in the below.

| Dataset    | 1/16 | 1/8  | 1/4  | 1/2  |
| ---------- | ---- | ---- | ---- | ---- |
| VOC        | 32   | 34   | 40   | 60   |
| Cityscapes | 128  | 137  | 160  | 240  |

