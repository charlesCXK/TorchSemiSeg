# TorchSemiSeg
[CVPR 2021] Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision
- High performance, simple framework.

## Installation

The code is developed using Python 3.6 with PyTorch 1.0.0. The code is developed and tested using 4 or 8 Tesla V100 GPUs.

1. **Clone this repo.**

   ```shell
   $ git clone https://github.com/charlesCXK/TorchSemiSeg.git
   $ cd TorchSemiSeg
   ```

2. **Install dependencies.**

   **(1) Create a conda environment:**

   ```shell
   $ conda env create -f semiseg.yaml
   $ conda activate semiseg
   ```

   **(2) Install apex 0.1(needs CUDA)**

   ```shell
   $ cd ./furnace/apex
   $ python setup.py install --cpp_ext --cuda_ext
   ```

## Training and Inference

##### Training && Inference on PASCAL VOC:

```shell
$ cd ./model/voc8.res50v3+.CPS
$ bash script.sh
```

- The tensorboard file is saved in `log/tb/` directory.
- In `script.sh`, you need to specify some variables, such as the path to your data dir, the path to your snopshot dir that stores checkpoints, etc.

## Citation

Please consider citing this project in your publications if it helps your research.

```tex
@inproceedings{chen2021-CPS,
  title={Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision},
  author={Chen, Xiaokang and Yuan, Yuhui and Zeng, Gang and Wang, Jingdong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

#### TODO
- [ ] Dataset release
- [ ] Code for Cityscapes dataset
- [ ] Code for CPS + CutMix
- [ ] Other SOTA semi-supervised segmentation methods
