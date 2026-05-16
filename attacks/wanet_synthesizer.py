from PIL import Image
import torchvision.transforms as transforms
from .synthesizer import Synthesizer
import torch.nn.functional as F
from tasks.task import Task
import torch
import os

class WaNetParameters():
    def __init__(self):
        # self.dataset = 'cifar10'
        self.input_height = 32
        self.input_width = 32
        self.input_channel = 3
        self.attack_mode = "all2one"
        
        self.checkpoints = "./synthesizers/triggers/wanet_checkpoints"
        self.ckpt_folder = None
        
        self.s = 0.5
        self.k = 4
        self.grid_rescale = 1.0
        self.device = None

class WaNetSynthesizer(Synthesizer):
    def __init__(self, task: Task, dataset: str):
        super().__init__(task)
        self.img_size = task.params.input_shape  # 例如 (3, 32, 32) 或 (3, 224, 224)
        self.task = task
        self.opt = WaNetParameters()
        self.opt.device = task.params.device
        self.dataset = dataset
        self.opt.input_channel = self.img_size[0]
        self.opt.input_height = self.img_size[1]
        self.opt.input_width = self.img_size[2]
        self.opt.dataset = dataset
        self.identity_grid = None
        self.noise_grid = None
        
        self.identity_grid, self.noise_grid = self.get_Generator(dataset)
    
    def get_Generator(self, dataset):
        self.opt.ckpt_folder = os.path.join(self.opt.checkpoints, self.opt.dataset)
        self.opt.ckpt_path = os.path.join(self.opt.ckpt_folder, "{}_{}_morph.pth.tar".format(self.opt.dataset, self.opt.attack_mode))
        self.opt.log_dir = os.path.join(self.opt.ckpt_folder, "log_dir")
        
        state_dict = torch.load(self.opt.ckpt_path)
        identity_grid = state_dict["identity_grid"]
        noise_grid = state_dict["noise_grid"]

        return identity_grid, noise_grid
        
    def synthesize_inputs(self, batch, attack_portion=None):
        
        grid_temps = (self.identity_grid + self.opt.s * self.noise_grid / self.opt.input_height) * self.opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(batch.batch_size, self.opt.input_height, self.opt.input_height, 2).to(self.opt.device) * 2 - 1
        grid_temps2 = grid_temps.repeat(batch.batch_size, 1, 1, 1) + ins / self.opt.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
        
        batch.inputs[:attack_portion] = F.grid_sample(batch.inputs[:attack_portion], grid_temps.repeat(batch.batch_size, 1, 1, 1), align_corners=True)
        return

    def synthesize_labels(self, batch, attack_portion=None):
        # 可根据需要在这里修改标签
        return
    
    
    def apply_backdoor_to_a_sample(self, data, label, params):
        data = data.unsqueeze(0)  # 添加 batch 维度，变为 (1, channels, height, width)
        grid_temps = (self.identity_grid + self.opt.s * self.noise_grid / self.opt.input_height) * self.opt.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(1, self.opt.input_height, self.opt.input_height, 2).to(self.opt.device) * 2 - 1
        grid_temps2 = grid_temps.repeat(1, 1, 1, 1) + ins / self.opt.input_height
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)
        
        backdoor_sample = F.grid_sample(data, grid_temps.repeat(1, 1, 1, 1), align_corners=True)
        backdoor_sample = backdoor_sample.squeeze(0)  # 去掉 batch 维度
        
        return backdoor_sample
