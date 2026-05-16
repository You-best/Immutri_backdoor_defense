from PIL import Image
import torchvision.transforms as transforms
from .synthesizer import Synthesizer
from tasks.task import Task
import torch

class BlendSynthesizer(Synthesizer):
    def __init__(self, task: Task, alpha: float = 0.5):
        super().__init__(task)
        self.img_size = task.params.input_shape  # 例如 (3, 32, 32) 或 (3, 224, 224)
        self.task = task
        self.alpha = alpha
        self.blend_tensor = None  # 初始化为 None
        self.get_blend_img()
    
    def synthesize_inputs(self, batch, attack_portion=None):
        batch.inputs[:attack_portion] = (1 - self.alpha) * \
                                        batch.inputs[:attack_portion] + \
                                        self.alpha * self.blend_tensor
        return

    def synthesize_labels(self, batch, attack_portion=None):
        # 可根据需要在这里修改标签
        return
    
    def get_blend_img(self):
        if self.img_size[1] == 32:
            img_path = '/home/star/sda1/backdoors101-master/synthesizers/triggers/hellokitty_32.png'
        elif self.img_size[1] == 224:
            img_path = '/home/star/sda1/backdoors101-master/synthesizers/triggers/hellokitty_224.png'
        else:
            raise ValueError(f"not support this image size: {self.img_size}")
        
        # 加载图像并转换为tensor
        img = Image.open(img_path).convert('RGB')  # 保证图片为RGB模式
        transform = transforms.Compose([
            transforms.ToTensor()  # 将图片转换为Tensor
        ])
        self.blend_tensor = transform(img)  # 3*32*32 或 3*224*224 的 Tensor
        self.blend_tensor = self.blend_tensor.to(self.task.params.device)
        # 返回转换后的Tensor
        return self.blend_tensor
    
    def apply_backdoor_to_a_sample(self, data, label, params):
        if self.blend_tensor is None:
            self.get_blend_img()  # 确保blend_tensor已加载
        backdoor_sample = (1 - self.alpha) * data + self.alpha * self.blend_tensor
        return backdoor_sample
