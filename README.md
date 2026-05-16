# IMMUTRI: Immunity-enabled Tri-Model Defense System

## Abstract

Motivated by the observation that backdoor attacks exploit smooth loss surfaces in low-dimensional trigger subspaces, we propose **IMMUTRI**, a two-stage post-training white-box defense for single-target data-poisoning backdoors. 

**Stage 1 - Loss-Guided Feature Decoupling (LGFD):** Selects a conservative first-minimum loss operating point, infers the likely attacked target label, and refines the resulting candidate set through class-conditional feature matching.

**Stage 2 - Tri-Model Functional Synergy (TMFS):** Fine-tunes a copy of the delivered model with fresh uniform label noise to obtain an auxiliary detector, uses it to purify the training set, retrains a reference model on the retained data, and deploys a tri-model firewall.

All expensive processing is offline; online inference requires at most three forward passes per query. The framework jointly outputs a purified retained training subset for repair and a deployment-time reject-and-answer mechanism without requiring trusted clean data or trigger reconstruction.

Evaluated against five attacks on CIFAR10/ImageNet-10/GTSRB, this functional synergy achieves near-perfect immunity with exceptional isolation capability while suppressing the average Attack Success Rate (ASR) and elevating Clean Accuracy (CA).

## Method Overview

### Stage 1: Loss-Guided Feature Decoupling (LGFD)
1. **Loss Spectrum Analysis**: Compute per-sample losses and detect the first minimum operating point
2. **Target Label Inference**: Analyze prediction patterns on suspicious samples to infer the attack target
3. **Directional Feature Matching**: Refine candidates using class-conditional feature alignment with the backdoor direction

### Stage 2: Tri-Model Functional Synergy (TMFS)
1. **Auxiliary Detector Training**: Fine-tune a model copy with uniform label noise to create a backdoor-sensitive detector
2. **Dataset Purification**: Use the detector to identify and retain clean samples based on loss thresholds
3. **Reference Model Training**: Train a clean reference model on the purified dataset
4. **Tri-Model Firewall Deployment**: Deploy three models (original, detector, reference) for consensus-based inference 


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
```bash
python attack.py --dataset cifar10 --attack badnets --batch_size 64 --gpu 0
```
The trained backdoor model will be saved in the `save` folder.

#### 2. Deploy IMMUTRI Defense

Apply the two-stage IMMUTRI defense to a backdoored model:
```bash
python ImmuTri.py --dataset cifar10 --attack badnets --batch_size 64 --gpu 0
```

This will execute:
- **LGFD Stage**: Analyze loss distribution, infer target label, and refine suspicious samples
- **TMFS Stage**: Train auxiliary detector, purify dataset, train reference model, and deploy tri-model firewall

The following models will be saved in the `save` folder:
- `{dataset}_{attack}_backdoor_model.pth`: Original delivered model (fbd)
- `{dataset}_{attack}_auxiliary_detector.pth`: Auxiliary detector trained with uniform noise (fenh)
- `{dataset}_{attack}_reference_model.pth`: Reference model trained on purified data (fref)
## Citation

```

If you find our work useful for your research, please consider citing our paper:

```

## 项目声明 Project Statement
本项目的作者及单位
The anthor and affiliation of this project
```

项目名称（Project Name）：Immutri_backdoor_defense
项目作者（Author）：Zaobo He，Jin Wan, Yusen Li，jie Zhong
作者单位（Affiliation）：暨南大学网络空间安全学院（College of Cyber Security，Jinan University）

```

```
