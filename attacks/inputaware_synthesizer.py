import os
from PIL import Image
import torchvision.transforms as transforms
from .synthesizer import Synthesizer
from tasks.task import Task
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms


class Conv2dBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, ker_size, stride, padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=ker_size, stride=stride, dilation=dilation, ceil_mode=ceil_mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, scale_factor=(2, 2), mode="bilinear", p=0.0):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


# ---------------------------- Generators ----------------------------#


class Generator(nn.Sequential):
    def __init__(self, opt, out_channels=None):
        super(Generator, self).__init__()
        if opt.dataset == "mnist":
            channel_init = 16
            steps = 2
        else:
            channel_init = 32
            steps = 3

        channel_current = opt.input_channel
        channel_next = channel_init
        for step in range(steps):
            self.add_module("convblock_down_{}".format(2 * step), Conv2dBlock(channel_current, channel_next))
            self.add_module("convblock_down_{}".format(2 * step + 1), Conv2dBlock(channel_next, channel_next))
            self.add_module("downsample_{}".format(step), DownSampleBlock())
            if step < steps - 1:
                channel_current = channel_next
                channel_next *= 2

        self.add_module("convblock_middle", Conv2dBlock(channel_next, channel_next))

        channel_current = channel_next
        channel_next = channel_current // 2
        for step in range(steps):
            self.add_module("upsample_{}".format(step), UpSampleBlock())
            self.add_module("convblock_up_{}".format(2 * step), Conv2dBlock(channel_current, channel_current))
            if step == steps - 1:
                self.add_module(
                    "convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next, relu=False)
                )
            else:
                self.add_module("convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next))
            channel_current = channel_next
            channel_next = channel_next // 2
            if step == steps - 2:
                if out_channels is None:
                    channel_next = opt.input_channel
                else:
                    channel_next = out_channels

        self._EPSILON = 1e-7
        self._normalizer = self._get_normalize(opt)
        self._denormalizer = self._get_denormalize(opt)

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def forward(self, x):
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x) / (2 + self._EPSILON) + 0.5
        return x

    def normalize_pattern(self, x):
        if self._normalizer:
            x = self._normalizer(x)
        return x

    def denormalize_pattern(self, x):
        if self._denormalizer:
            x = self._denormalizer(x)
        return x

    def threshold(self, x):
        return nn.Tanh()(x * 20 - 10) / (2 + self._EPSILON) + 0.5

class InputAwareParameters():
    def __init__(self):
        self.dataset = 'cifar10'
        self.input_height = 32
        self.input_width = 32
        self.input_channel = 3
        
    
class InputAwareSynthesizer(Synthesizer):
    def __init__(self, task: Task, dataset: str):
        super().__init__(task)
        self.img_size = task.params.input_shape  # 例如 (3, 32, 32) 或 (3, 224, 224)
        self.task = task
        self.netG = None
        self.netM = None
        self.opt = InputAwareParameters()
        self.netG, self.netM = self.get_Generator(dataset)
    
    def get_Generator(self, dataset):
        path_model = os.path.join(
            "./synthesizers/triggers/input_aware_checkpoints", dataset, "all2one", "{}_{}_ckpt.pth.tar".format("all2one", dataset)
        )
        state_dict = torch.load(path_model)
        
        self.opt.dataset = dataset
        self.opt.input_channel = self.img_size[0]
        self.opt.input_height = self.img_size[1]
        self.opt.input_width = self.img_size[2]
        
        netG = Generator(self.opt)
        netG.load_state_dict(state_dict["netG"])
        netG.to(self.task.params.device)
        netG.eval()
        netG.requires_grad_(False)
        
        netM = Generator(self.opt, out_channels=1)
        netM.load_state_dict(state_dict["netM"])
        netM.to(self.task.params.device)
        netM.eval()
        netM.requires_grad_(False)
        return netG, netM
    
    def synthesize_inputs(self, batch, attack_portion=None):
        # 未实现
        return

    def synthesize_labels(self, batch, attack_portion=None):
        # 可根据需要在这里修改标签
        return
    
    
    def apply_backdoor_to_a_sample(self, data, label, params):
        # 将 data 变为 4 维 (batch_size, channels, height, width)
        data = data.unsqueeze(0)  # 添加 batch 维度，变为 (1, channels, height, width)

        # 生成 pattern 并应用 normalization
        patterns = self.netG(data)
        patterns = self.netG.normalize_pattern(patterns)

        # 使用 mask 并生成 backdoor 样本
        masks_output = self.netM.threshold(self.netM(data))
        backdoor_sample = data + (patterns - data) * masks_output

        # 将 backdoor_sample 变回三维 (channels, height, width)
        backdoor_sample = backdoor_sample.squeeze(0)  # 去掉 batch 维度

        return backdoor_sample

