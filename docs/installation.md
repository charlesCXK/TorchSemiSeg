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
  
## Optional
We recommend using docker to run experiments. Here is the docker name: charlescxk/ssc:2.0 .
You could pull it from https://hub.docker.com/ and mount your local file system to it.
