import random
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import transforms, functional
import os
from .synthesizer import Synthesizer
from tasks.task import Task
import torch.nn.functional as F
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
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
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
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
    def __init__(
        self, block, num_blocks, num_classes=10, in_channel=3, zero_init_residual=False
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class Attack(object):
    def __init__(self, name, model):
        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device

        self._targeted = 1
        self._attack_mode = "original"
        self._return_type = "float"

    def forward(self, *input):
        raise NotImplementedError

    def set_attack_mode(self, mode):
        if self._attack_mode == "only_original":
            raise ValueError(
                "Changing attack mode is not supported in this attack method."
            )

        if mode == "original":
            self._attack_mode = "original"
            self._targeted = 1
            self._transform_label = self._get_label
        elif mode == "targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._transform_label = self._get_label
        elif mode == "least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(
                mode
                + " is not a valid mode. [Options : original, targeted, least_likely]"
            )

    def set_return_type(self, type):
        if type == "float":
            self._return_type = "float"
        elif type == "int":
            self._return_type = "int"
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, save_path, data_loader, verbose=True):
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == "int":
                adv_images = adv_images.float() / 255

            if verbose:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print(
                    "- Save Progress : %2.2f %% / Accuracy : %2.2f %%"
                    % ((step + 1) / total_batch * 100, acc),
                    end="\r",
                )

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), save_path)
        print("\n- Save Complete!")

        self._switch_model()

    def _transform_label(self, images, labels):
        return labels

    def _get_label(self, images, labels):
        return labels

    def _get_least_likely_label(self, images, labels):
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels

    def _to_uint(self, images):
        return (images * 255).type(torch.uint8)

    def _switch_model(self):
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ["model", "attack"]

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info["attack_mode"] = self._attack_mode
        if info["attack_mode"] == "only_original":
            info["attack_mode"] = "original"

        info["return_type"] = self._return_type

        return (
            self.attack
            + "("
            + ", ".join("{}={}".format(key, val) for key, val in info.items())
            + ")"
        )

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == "int":
            images = self._to_uint(images)

        return images
    
class PGD(Attack):
    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        # 如果 labels 是一个整数，将其转换为张量
        if isinstance(labels, int):
            labels = torch.tensor([labels], dtype=torch.long, device=self.device)
        else:
            labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class CLSynthesizer(Synthesizer):
    def __init__(self, task: Task, dataset='cifar10', model_path="./synthesizers/triggers/CL_adv_models"):
        super().__init__(task)
        self.task = task
        self.device = task.params.device
        self.img_size = task.params.input_shape  # 例如 (3, 32, 32) 或 (3, 224, 224)
        self.adv_model = self.get_adv_model(model_path, dataset)
        self.dataset = dataset
        self.pgd_config = self.set_pgd_config()
        self.attacker = PGD(self.adv_model, eps=self.pgd_config['eps'], alpha=self.pgd_config['alpha'], steps=self.pgd_config['steps'])
        self.attacker.set_return_type("int")
        self.trigger_ptn = None
        self.trigger_loc = None
        self.get_trigger()
        
        
    def get_trigger(self, trigger_path="./synthesizers/triggers/cifar_1.png"):
        with open(trigger_path, "rb") as f:
            trigger_ptn = Image.open(f).convert("RGB")

        # 将图片转换为 NumPy 数组并再转换为 PyTorch Tensor
        self.trigger_ptn = np.array(trigger_ptn)
        # 找到非零元素的位置，并将其转换为 Tensor
        self.trigger_loc = np.nonzero(self.trigger_ptn)
        self.trigger_ptn = torch.tensor(self.trigger_ptn).permute(2, 0, 1) * 0.1  # CHW
        self.trigger_loc = [torch.tensor(loc, dtype=torch.long) for loc in self.trigger_loc]
        # 创建一个和 adv_sample 一样形状的掩码，用来指定触发器的位置
        mask = torch.zeros([self.img_size[1], self.img_size[2]])
        mask[self.trigger_loc[0], self.trigger_loc[1]] = 1
        self.mask = mask.unsqueeze(0).repeat(self.img_size[0], 1, 1)
        
        
    def set_pgd_config(self, eps=8, alpha=1.5, step=100, max_pixel=255):
        pgd_config = { 'eps': eps, 'alpha': alpha, 'steps': step, 'max_pixel': max_pixel}
        max_pixel = pgd_config['max_pixel']
        for k, v in pgd_config.items():
            if k == "eps" or k == "alpha":
                pgd_config[k] = v / max_pixel
        return pgd_config
    
    def get_adv_model(self, model_path, dataset):
        if dataset=='cifar10':
            model_path = os.path.join(model_path, "cifar_resnet_e8_a2_s10.pth")
        else:
            print("暂时不支持该数据集")
            exit()
        adv_model = resnet18()
        adv_ckpt = torch.load(model_path)
        adv_model.load_state_dict(adv_ckpt)
        adv_model = adv_model.to(self.task.params.device)
        return adv_model
    
    def synthesize_inputs(self, batch, attack_portion=None):
        
        return

    def synthesize_labels(self, batch, attack_portion=None):
        # batch.labels[:attack_portion].fill_(self.params.backdoor_label)

        return
    
    def apply_backdoor_to_a_sample(self, data, label, params):
        # 如果 data 是 3 维的 [C, H, W]，则增加一个 batch 维度变成 [1, C, H, W]
        if data.dim() == 3:
            data = data.unsqueeze(0)  # 增加 batch 维度
        # 将触发器模式和位置移动到与输入数据相同的设备上
        self.trigger_ptn = self.trigger_ptn.to(data.device)
        if params['adv']:
            # 确保 data 是 4 维的 [B, C, H, W]
            adv_sample = self.attacker.forward(data, label)
        else:
            adv_sample = data
        # adv_sample = self.task.denormalize(adv_sample)
        # 将 adv_sample 的触发器位置置 0
        mask = self.mask.repeat(data.size(0), 1, 1, 1).to(data.device)
        adv_sample[mask == 1] = 0

        # 将触发器图案应用到指定位置
        backdoor_sample = adv_sample + self.trigger_ptn * mask  # 仅在掩码处应用触发器
        
        # backdoor_sample = self.task.normalize(backdoor_sample)
        # 如果输入的 data 原本是 3 维的，则去掉增加的 batch 维度
        if data.size(0) == 1:
            backdoor_sample = backdoor_sample.squeeze(0)  # 去掉 batch 维度

        return backdoor_sample

