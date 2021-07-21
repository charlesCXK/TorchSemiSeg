## Getting Started

### Data Preparation 
##### Download the data (VOC, Cityscapes) and pre-trained models from  [OneDrive link](https://pkueducn-my.sharepoint.com/:f:/g/personal/pkucxk_pku_edu_cn/EtjNKU0oVMhPkOKf9HTPlVsBIHYbACel6LSvcUeP4MXWVg?e=139icd): 

```
DATA/
|-- city
|-- pascal_voc
|-- pytorch-weight
|   |-- resnet50_v1c.pth
|   |-- resnet101_v1c.pth
```


### Training && Inference on PASCAL VOC:

```shell
$ cd ./model/voc8.res50v3+.CPS
$ bash script.sh
```

- The tensorboard file is saved in `log/tb/` directory.
- In `script.sh`, you need to specify some variables, such as the path to your data dir, the path to your snapshot dir that stores checkpoints, etc.
- We have released the [training log](https://pkueducn-my.sharepoint.com/:t:/g/personal/pkucxk_pku_edu_cn/ERl1pm99zhFIvIIB8y82WiIB13AQ9Hd8FyrJQ5v3fpP0vg?e=Nd0f1M) and [pretrained model](https://pkueducn-my.sharepoint.com/:u:/g/personal/pkucxk_pku_edu_cn/ESx6vF1dapJGkbsLWHnHSakBdFOkooQcIFeDpCRTVJS8Iw?e=zMCvhj) for this experiment on OneDrive. The performance is slightly different (73.28) from that of paper (73.20) due to randomness.
- We have also released the [training log](https://pkueducn-my.sharepoint.com/:t:/g/personal/pkucxk_pku_edu_cn/EdRRCsS2KtFGoTophitkLh0BnA40ZPBmuVKhWEV-biF2lw?e=8LaE88) of `city8.res50v3+.CPS`.

### Different Partitions
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

