# Immunity against Backdoor

## Abstract

The reliance of deep neural networks (DNNs) on third-party data and cloud-based training exacerbates stealthy backdoor risks, while current defenses face three limitations: poor generalization to complex triggers, dependency on clean data subsets, and prohibitive costs for trigger reverse engineering.In this paper, we experimentally observe that backdoor attacks exploit smooth loss surfaces in low-dimensional trigger subspaces, inducing rapid model overfitting and resulting in a cross-scale loss gap between poisoned and clean samples. Building on this observation, we propose \textbf{Immunity-enabled Tri-Model (ImmuTri)} Defense System, a robust inference-time framework integrating three core techniques:  Loss-Guided Feature Decoupling isolates suspicious samples via dynamic thresholds and identifies target classes through manifold shifts; Function Perturbation-Driven Purification amplifies loss discrepancies via label noise to train backdoor-sensitive detectors; and a cascaded firewall synergizes original, backdoor-sensitive, and mildly poisoned models for real-time blocking. Evaluated against five attacks on CIFAR10/ImageNet-10/GTSRB, this functional synergy achieves near-perfect immunity:  On GTSRB, it attains $100.00\%$ True Positive Rate (TPR) and $1.79\%$ False Positive Rate (FPR), while simultaneously demonstrating exceptional isolation capability with $\mathbf{98.22\%}$ True Negative Rate (TNR) and $\mathbf{0.00\%}$ False Negative Rate (FNR). The framework suppresses the average Attack Success Rate (ASR) from $92.50\%$ to 1.32\% and elevates Clean Accuracy (CA) to $94.57\%$. 


## Getting Started

### Dataset

#### Download the Datasets

Download the pre-processed datasets and unzip them into the `data` folder. The final directory structure should look like `data/CIFAR10`, `data/GTSRB`, etc.

  * **CIFAR-10**

      * Download the pre-processed dataset `CIFAR10.tar.gz` from this [link](https://drive.google.com/file/d/1vZtzN3a33rKitnqz5ad9S8v5Rv6CK2-Q/view?usp=sharing), and unzip it into the `data` folder.

  * **GTSRB (German Traffic Sign Recognition Benchmark)**

      * Download the dataset images from the official website: [GTSRB\_Final\_Training\_Images.zip](https://www.google.com/search?q=https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip).
      * You will also need the test set annotations: [GTSRB\_Final\_Test\_GT.zip](https://www.google.com/search?q=https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip).
      * Unzip the training images and organize them into a `data/GTSRB` folder.

  * **ImageNet10**

      * ImageNet10 is a small 10-class subset of the full ImageNet dataset, making it much faster to use for testing and development.
      * Download the pre-processed dataset `ImageNet10.tar.gz` from this [link](https://www.google.com/search?q=https://drive.google.com/file/d/1q-a_gcsRUmv3wX4yJjmm22y4o3yH527f/view%3Fusp%3Dsharing), and unzip it into the `data` folder.


### Usage

#### 1. Train Backdoored Models

Train a backdoored model on CIFAR10 for BadNets attack:
```
python attack.py --dataset cifar10 --attack badnets --batch_size 6
4 --gpu 0
```
The results of attack can be found in folder `attack`.


#### 2. Utilize ImmuTri for Defense

Defend a backdoored model on CIFAR10 for BadNets attack:
```
python ImmuTri.py --dataset cifar10 --attack badnets --batch_size 
64 --gpu 0
```
## Citation

```

If you find our work useful for your research, please consider citing our paper:

```

## The anthor and affiliation of this project
```

项目名称（Project Name）：Immutri_backdoor_defense
项目作者（Author）：Zaobo He，Jin Wan, Yusen Li，jie Zhong
作者单位（Affiliation）：暨南大学网络空间安全学院（College of Cyber Security，Jinan University）

```

```
