import torch.nn as nn
import torch
import torch.nn.functional as F

class Model(nn.Module):
    """
    Base class for models with added support for GradCam activation map
    and a SentiNet defense. The GradCam design is taken from:
https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    If you are not planning to utilize SentiNet defense just import any model
    you like for your tasks.
    """

    def __init__(self):
        super().__init__()
        self.gradient = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def switch_grads(self, enable=True):
        for i, n in self.named_parameters():
                n.requires_grad_(enable)

    def features(self, x):
        """
        Get latent representation, eg logit layer.
        :param x:
        :return:
        """
        raise NotImplemented

    def forward(self, x, latent=False):
        raise NotImplemented

class SCELoss(nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, pred, labels, reduction='mean'):
        # CCE (计算每个样本的单独损失)
        ce = F.cross_entropy(pred, labels, reduction='none')  # 不聚合损失值
        
        # RCE (逆交叉熵)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))  # 不聚合RCE损失

        # 计算每个样本的总损失
        loss = self.alpha * ce + self.beta * rce

        # 根据 reduction 参数决定是返回每个样本的损失，还是计算均值
        if reduction == 'mean':
            return loss.mean()  # 返回均值损失
        elif reduction == 'sum':
            return loss.sum()  # 返回总和损失
        else:  # reduction == 'none'
            return loss  # 返回每个样本的损失值
        
class CrossEntropyLoss_soft(nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss_soft, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        # Apply log_softmax to the input logits
        log_probs = F.log_softmax(output, dim=-1)
        
        # Compute the element-wise multiplication between log_probs and the target (soft labels)
        loss = -torch.sum(target * log_probs, dim=-1)
        
        # Return the mean or sum depending on the reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss