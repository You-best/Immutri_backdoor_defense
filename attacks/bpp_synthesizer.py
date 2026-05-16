import os
import torch
import torch.nn.functional as F
from .synthesizer import Synthesizer
from tasks.task import Task
import random
from numba import jit
from numba.types import float64, int64
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

class BppParameters:
    def __init__(
        self,
        data_root="/home/star/sda1/data/1744ecab-59b5-4839-9124-a5d97b52e660/datasets/Imagenet10",
        checkpoints="./synthesizers/triggers/bpp_checkpoints",
        temps="./temps",
        device="cuda",
        continue_training=False,
        model_filepath="./checkpoints",
        dataset="imagenet10",
        set_arch=None,
        attack_mode="all2one",
        save_all=False,
        save_freq=50,
        bs=32,
        lr=1e-2,
        scheduler_milestones=[100, 200, 300, 400],
        scheduler_lambda=0.1,
        n_iters=1000,
        num_workers=6,
        target_label=0,
        injection_rate=0.2,
        neg_rate=0.2,
        random_rotation=10,
        random_crop=5,
        squeeze_num=32,
        dithering=False
    ):
        self.data_root = data_root
        self.checkpoints = checkpoints
        self.temps = temps
        self.device = device
        self.continue_training = continue_training
        self.model_filepath = model_filepath
        self.dataset = dataset
        self.set_arch = set_arch
        self.attack_mode = attack_mode
        self.save_all = save_all
        self.save_freq = save_freq
        self.bs = bs
        self.lr = lr
        self.scheduler_milestones = scheduler_milestones
        self.scheduler_lambda = scheduler_lambda
        self.n_iters = n_iters
        self.num_workers = num_workers
        self.target_label = target_label
        self.injection_rate = injection_rate
        self.neg_rate = neg_rate
        self.random_rotation = random_rotation
        self.random_crop = random_crop
        self.squeeze_num = squeeze_num
        self.dithering = dithering

        self.input_channel = 3
        self.input_height = 224
        self.input_width = 224

        # 可选路径派生配置（类似 WaNet 中的 ckpt_path）
        self.ckpt_folder = os.path.join(self.checkpoints, self.dataset)
        self.ckpt_path = os.path.join(
            self.ckpt_folder,
            f"{self.dataset}_{self.attack_mode}_bpp.pth.tar"
        )
        self.log_dir = os.path.join(self.ckpt_folder, 'log_dir')

@jit(float64[:](float64[:], int64, float64[:]),nopython=True)
def rnd1(x, decimals, out):
    return np.round(x, decimals, out)

@jit(nopython=True)
def floydDitherspeed(image,squeeze_num):
    channel, h, w = image.shape
    for y in range(h):
        for x in range(w):
            old = image[:,y, x]
            temp=np.empty_like(old).astype(np.float64)
            new = rnd1(old/255.0*(squeeze_num-1),0,temp)/(squeeze_num-1)*255
            error = old - new
            image[:,y, x] = new
            if x + 1 < w:
                image[:,y, x + 1] += error * 0.4375
            if (y + 1 < h) and (x + 1 < w):
                image[:,y + 1, x + 1] += error * 0.0625
            if y + 1 < h:
                image[:,y + 1, x] += error * 0.3125
            if (x - 1 >= 0) and (y + 1 < h): 
                image[:,y + 1, x - 1] += error * 0.1875
    return image

import csv
class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []

    # Common initial transform: Resize
    # For ImageNet, if original images are larger, this resize is usually to a slightly
    # larger size than opt.input_height/width before RandomResizedCrop or CenterCrop.
    # E.g., Resize(256) then RandomResizedCrop(224) or CenterCrop(224).
    # Your current code resizes directly to opt.input_height/width. This is fine if that's intended.
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))

    if pretensor_transform: # This flag seems to control a specific set of augmentations
        if train:
            # Your existing pre-tensor augmentations
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))

            # Dataset-specific augmentations before ToTensor()
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
            elif opt.dataset == "imagenet10":
                # Common ImageNet augmentations (if not already covered by general ones)
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
                # RandomResizedCrop is very common for ImageNet training.
                # If you use this, you might adjust the initial Resize.
                # Example:
                # transforms_list.insert(0, transforms.RandomResizedCrop(opt.input_height))
                # And remove the general Resize and RandomCrop if RandomResizedCrop is preferred.
                # For now, I'll just add RandomHorizontalFlip as it's generally beneficial.
                # Your existing RandomCrop will apply, but RandomResizedCrop is often stronger for ImageNet.

    # Convert to Tensor (moves image to [0,1] range, C x H x W)
    transforms_list.append(transforms.ToTensor())

    # Normalization (applied after ToTensor)
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "imagenet10": # Added ImageNet10 normalization
        transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
        # For GTSRB and CelebA, you are currently not applying any normalization.
        # This means your model will expect input in the [0, 1] range for these datasets.
        # This is fine if intended, but often normalization is beneficial.
        # If they were already in [0,1] and you wanted to keep them that way, 'pass' is correct.
        # If they are [0,255] PIL images and ToTensor() makes them [0,1], and no further normalization
        # is needed, then 'pass' is also correct.
        pass
    else:
        raise Exception(f"Invalid Dataset for normalization: {opt.dataset}")

    return transforms.Compose(transforms_list)


def get_dataloader(opt, train=True, pretensor_transform=False, shuffle=True):
    transform = get_transform(opt, train, pretensor_transform)

    if opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform=transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=transform, download=True)
    elif opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "imagenet10": # 新增对 imagenet10 的处理
        # 确定是加载训练集还是验证/测试集
        # 假设你的 Imagenet10 数据集在 opt.data_root 下有 'train' 和 'val' (或 'test') 子目录
        split_folder = "train" if train else "val" # 你可能需要根据实际情况调整 'val' 或 'test'

        # 构建到特定数据集分割的完整路径
        # opt.data_root 指向 ".../datasets/Imagenet10"
        # 我们期望的结构是 ".../datasets/Imagenet10/train" 和 ".../datasets/Imagenet10/val"
        dataset_path = os.path.join(opt.data_root, split_folder)

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(
                f"ImageNet10 data directory for split '{split_folder}' not found at: {dataset_path}. "
                f"Please ensure your ImageNet10 dataset is structured as: {opt.data_root}/<split_folder>/<class_name>/<image_files>"
            )

        dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)    
    
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=shuffle)
    return dataloader, transform




def back_to_np_4d(inputs, opt):
    if opt.dataset == "cifar10":
        expected_values = [0.4914, 0.4822, 0.4465]
        variance = [0.247, 0.243, 0.261] # These are standard deviations
    elif opt.dataset == "mnist":
        expected_values = [0.5]
        variance = [0.5] # Standard deviation
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba": # Combined as they use same defaults
        expected_values = [0.0, 0.0, 0.0]
        variance = [1.0, 1.0, 1.0] # Std deviations (means no normalization was applied or data is [0,1])
    elif opt.dataset == "imagenet10": # Added imagenet10
        expected_values = [0.485, 0.456, 0.406] # Standard ImageNet means
        variance = [0.229, 0.224, 0.225]      # Standard ImageNet stds
    else:
        raise ValueError(f"Unsupported dataset for back_to_np_4d: {opt.dataset}")

    # inputs_clone = inputs.clone()
    # inputs_clone = inputs.clone().cpu()
    inputs_clone = inputs.detach().cpu().clone()

    if opt.dataset == "mnist":
        # De-normalize: (normalized_value * std) + mean
        inputs_clone[:, 0, :, :] = inputs_clone[:, 0, :, :] * variance[0] + expected_values[0]
    else: # For cifar10, gtsrb, celeba, imagenet10 (all 3-channel)
        for channel in range(opt.input_channel): # Use opt.input_channel for robustness
            inputs_clone[:, channel, :, :] = inputs_clone[:, channel, :, :] * variance[channel] + expected_values[channel]

    # Scale to [0, 255]
    return inputs_clone * 255.0 # Ensure float multiplication if inputs_clone might be int

def build_residual_list(opt, train_dl, n_cycles=5):

    residual_list = []

    for _ in range(n_cycles):
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            # 1. 反归一化到 [0,255]
            #    back_to_np_4d 返回 shape=(B,C,H,W)，值域[0,255]
            temp = back_to_np_4d(inputs, opt)  # Tensor, CPU

            # 2. 量化
            if opt.dithering:
                # 抖动量化（Floyd–Steinberg）
                temp_mod = temp.clone()
                for i in range(temp_mod.shape[0]):
                    np_img = temp_mod[i].numpy()  # ndarray, float
                    dithered = floydDitherspeed(
                        np_img, 
                        float(opt.squeeze_num)
                    )  # ndarray, 已量化到 [0,255]
                    temp_mod[i] = torch.from_numpy(dithered).float()
            else:
                # 均匀量化
                # 先映射到 [0, squeeze_num-1]，四舍五入，再映射回 [0,255]
                temp_mod = torch.round(
                    temp / 255.0 * (opt.squeeze_num - 1)
                ) / (opt.squeeze_num - 1) * 255

            # 3. 计算残差
            residual = temp_mod - temp  # shape=(B,C,H,W), CPU

            # 4. 收集每张图的残差
            for i in range(residual.shape[0]):
                # unsqueeze 变成 (1,C,H,W)，detach 并转 CPU
                res_i = residual[i].unsqueeze(0).cpu().detach()
                residual_list.append(res_i)

    return residual_list
# opt = BppParameters()
# train_dl, _ = get_dataloader(opt, train=True)
# n_cycles = 1 if opt.dataset == "celeba" else 5
# residual_list_train = build_residual_list(opt, train_dl, n_cycles=n_cycles)
# synthesizer = BppSynthesizer(opt, residual_list_train)
from PIL import Image
class BppSynthesizer(Synthesizer):
    def __init__(self, task: Task, opt, residual_list_train, dataset):
        super().__init__(task)
        # Expect task.params.input_shape = (3, 224, 224)
        self.task = task
        self.opt = opt

        self.dataset = dataset
        self.squeeze_num = opt.squeeze_num
        self.residual_list_train = residual_list_train
        self.trigger_ptn = None
        self.mask        = None
        self.default_alpha = 0.3
        # self.get_trigger()

    def back_to_np_4d(self, inputs, opt):
        if opt.dataset == "cifar10":
            expected_values = [0.4914, 0.4822, 0.4465]
            variance = [0.247, 0.243, 0.261] # These are standard deviations
        elif opt.dataset == "mnist":
            expected_values = [0.5]
            variance = [0.5] # Standard deviation
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba": # Combined as they use same defaults
            expected_values = [0.0, 0.0, 0.0]
            variance = [1.0, 1.0, 1.0] # Std deviations (means no normalization was applied or data is [0,1])
        elif opt.dataset == "imagenet10": # Added imagenet10
            expected_values = [0.485, 0.456, 0.406] # Standard ImageNet means
            variance = [0.229, 0.224, 0.225]      # Standard ImageNet stds
        else:
            raise ValueError(f"Unsupported dataset for back_to_np_4d: {opt.dataset}")

        # inputs_clone = inputs.clone()
        # inputs_clone = inputs.clone().cpu()
        inputs_clone = inputs.detach().cpu().clone()

        if opt.dataset == "mnist":
            # De-normalize: (normalized_value * std) + mean
            inputs_clone[:, 0, :, :] = inputs_clone[:, 0, :, :] * variance[0] + expected_values[0]
        else: # For cifar10, gtsrb, celeba, imagenet10 (all 3-channel)
            for channel in range(opt.input_channel): # Use opt.input_channel for robustness
                inputs_clone[:, channel, :, :] = inputs_clone[:, channel, :, :] * variance[channel] + expected_values[channel]

        # Scale to [0, 255]
        return inputs_clone * 255.0 # Ensure float multiplication if inputs_clone might be int

    def np_4d_to_tensor(self, inputs_0_255, opt): # Renamed inputs to inputs_0_255 for clarity
        if opt.dataset == "cifar10":
            expected_values = [0.4914, 0.4822, 0.4465]
            variance = [0.247, 0.243, 0.261] # Standard deviations
        elif opt.dataset == "mnist":
            expected_values = [0.5]
            variance = [0.5] # Standard deviation
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba": # Combined
            expected_values = [0.0, 0.0, 0.0]
            variance = [1.0, 1.0, 1.0] # Std deviations
        elif opt.dataset == "imagenet10": # Added imagenet10
            expected_values = [0.485, 0.456, 0.406] # Standard ImageNet means
            variance = [0.229, 0.224, 0.225]      # Standard ImageNet stds
        else:
            raise ValueError(f"Unsupported dataset for np_4d_to_tensor: {opt.dataset}")

        # Scale from [0, 255] to [0, 1]
        inputs_clone = inputs_0_255.clone().div(255.0)

        if opt.dataset == "mnist":
            # Normalize: (value - mean) / std
            inputs_clone[:, 0, :, :] = (inputs_clone[:, 0, :, :] - expected_values[0]).div(variance[0])
        else: # For cifar10, gtsrb, celeba, imagenet10
            for channel in range(opt.input_channel): # Use opt.input_channel
                inputs_clone[:, channel, :, :] = (inputs_clone[:, channel, :, :] - expected_values[channel]).div(variance[channel])
        return inputs_clone  


    def synthesize_labels(self, batch, attack_portion=None):
        # keep original labels or redirect for all2one
        return batch.labels


    def apply_backdoor_to_a_sample(self, data, label, params=None):
        data = data.unsqueeze(0)  # (1, C, H, W)

        # 反归一化并处理
        np_img = self.back_to_np_4d(data, self.opt)
        if self.opt.dithering:
            np_img = torch.round(torch.from_numpy(floydDitherspeed(np_img.squeeze(0).numpy(), float(self.squeeze_num))))
            np_img = np_img.unsqueeze(0)
        else:
            np_img = torch.round(np_img / 255.0 * (self.squeeze_num - 1)) 
            np_img = np_img / (self.squeeze_num - 1) * 255

        # 转换为归一化 tensor
        backdoor_tensor = self.np_4d_to_tensor(np_img, self.opt).to(data.device)
        return backdoor_tensor.squeeze(0)


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512 * block.expansion * 4, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     return out
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)             # B×C×1×1
        out = out.view(out.size(0), -1)     # B×C
        out = self.linear(out)              # B×num_classes
        return out        
        
    def forward_activations(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        layer1 = self.layer1(out)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        out = F.avg_pool2d(layer4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return layer1,layer2,layer3,layer4,out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])




