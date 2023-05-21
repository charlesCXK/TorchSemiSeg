## Getting Started

### Data Preparation 
##### Download the data (VOC, Cityscapes) and pre-trained ResNet models from  [OneDrive link](https://pkueducn-my.sharepoint.com/:f:/g/personal/pkucxk_pkueducn_onmicrosoft_com/EtjNKU0oVMhPkOKf9HTPlVsBIHYbACel6LSvcUeP4MXWVg?e=tChnP7): 

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
- We have released the [training log](https://pkueducn-my.sharepoint.com/:f:/g/personal/pkucxk_pkueducn_onmicrosoft_com/EhONK8Ddq7pIqBJkgBAEJ2sBFeI7ZHN8PXCByA_WPTMZ1Q?e=gg6Rao) and [pretrained model](https://pkueducn-my.sharepoint.com/:f:/g/personal/pkucxk_pkueducn_onmicrosoft_com/EhONK8Ddq7pIqBJkgBAEJ2sBFeI7ZHN8PXCByA_WPTMZ1Q?e=gg6Rao) for this experiment on OneDrive. The performance is slightly different (73.28) from that of paper (73.20) due to randomness.
- We have also released the [training log](https://pkueducn-my.sharepoint.com/:f:/g/personal/pkucxk_pkueducn_onmicrosoft_com/Eg9iuGCF0idHsdb91jZkktIBKzChtamTeZnIvoL4DNbSsw?e=eZ7nB6) of `city8.res50v3+.CPS`.

### Different Partitions
To try other data partitions beside 1/8, you just need to change two variables in `config.py`:
```python
C.labeled_ratio = 8
C.nepochs = 34
```
Please note that, for fair comparison, we control the total iterations during training in each experiment similar (almost the same), including the supervised baseline and semi-supervised methods. Therefore, the nepochs for different partitions are different. 

We take VOC as an example.
1. We totally have 10582 images. The full supervised baseline is trained for 60 epochs with batch size 16, thus having 10582*60/16 = 39682.5 iters.
2. If we train CPS under the 1/8 split, we have 1323 labeled images and 9259 unlabeled images. Since the number of unlabeled images is larger than the number of labeled images, the `epoch` is defined as passing all the unlabeled images to the network. In each iteration, we have 8 labeled images and 8 unlabeled images, thus having 9259/8 = 1157.375 iters in one epoch. Then the total epochs we need is 39682.5/1157.375 = 34.29 ≈ 34. 
3. For the supervised baseline under the 1/8 split, the batch size 8 (as illustrated in Appendix A in the paper) and the iteration number is 39682.5 (the same as semi-supervised method).


We list the nepochs for different datasets and partitions in the below.

| Dataset    | 1/16 | 1/8  | 1/4  | 1/2  |
| ---------- | ---- | ---- | ---- | ---- |
| VOC        | 32   | 34   | 40   | 60   |
| Cityscapes | 128  | 137  | 160  | 240  |

