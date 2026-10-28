This training implementation is adapted from PyTorch’s official ImageNet example: `https://github.com/pytorch/examples/tree/main/imagenet`. For usage instructions, see `README_original.md` in this directory. *(Note: Original training code is licensed under the BSD 3-Clause License.)*

# Setup
Download the ImageNet1k (ILSVRC2012) dataset from the official website at https://image-net.org/. Note that access requires registration and acceptance of their terms. After downloading, extract the images into labeled `train/` and `val/` subfolders (e.g., using `extract_ILSVRC.sh`). The scripts below expect the dataset to be stored at `~/Datasets/imagenet`, structured as follows:
```
~/Datasets/imagenet/train/
~/Datasets/imagenet/val/
```
Install PyTorch:
```
pip install -r requirements.txt
```
We used two NVIDIA RTX 4090 24GB VRAM GPUs with Multi-processing Distributed Data Parallel Training in each experiment.
# Main results
```
./run_srevnet_imagenet.sh
```

# Additional MResNet momentum rate ablations
To train MResNet with a momentum value other than the default (0.9), run:
```
./run_srevnet_imagenet_mresnet.sh
```