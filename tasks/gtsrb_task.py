import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
import torch
from models.resnet_cifar import resnet18, resnet50, resnet34
from tasks.task import Task


class GtsrbTask(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    
    
    def load_data(self):
        self.load_cifar_data()
    
    
    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                self.normalize,
            ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.train_dataset = torchvision.datasets.GTSRB(
            root=self.params.data_path,
            split="train",
            download=True,
            transform=transform_train)
        if self.params.poison_images:
            self.train_loader = self.remove_semantic_backdoors()
        else:
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=0)
        self.test_dataset = torchvision.datasets.GTSRB(
            root=self.params.data_path,
            split="test",
            download=True,
            transform=transform_test)
        
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = (
            "Speed limit (20km/h)",
            "Speed limit (30km/h)",
            "Speed limit (50km/h)",
            "Speed limit (60km/h)",
            "Speed limit (70km/h)",
            "Speed limit (80km/h)",
            "End of speed limit (80km/h)",
            "Speed limit (100km/h)",
            "Speed limit (120km/h)",
            "No passing",
            "No passing for vehicles over 3.5 metric tons",
            "Right-of-way at the next intersection",
            "Priority road",
            "Yield",
            "Stop",
            "No vehicles",
            "Vehicles over 3.5 metric tons prohibited",
            "No entry",
            "General caution",
            "Dangerous curve to the left",
            "Dangerous curve to the right",
            "Double curve",
            "Bumpy road",
            "Slippery road",
            "Road narrows on the right",
            "Road work",
            "Traffic signals",
            "Pedestrians",
            "Children crossing",
            "Bicycles crossing",
            "Beware of ice/snow",
            "Wild animals crossing",
            "End of all speed and passing limits",
            "Turn right ahead",
            "Turn left ahead",
            "Ahead only",
            "Go straight or right",
            "Go straight or left",
            "Keep right",
            "Keep left",
            "Roundabout mandatory",
            "End of no passing",
            "End of no passing by vehicles over 3.5 metric tons"
        )
        return True

    def build_model(self, shallow=False) -> nn.Module:
        if self.params.pretrained:
            model = resnet18(pretrained=True, shallow_linear=shallow)

            # model is pretrained on ImageNet changing classes to CIFAR
            model.fc = nn.Linear(512, len(self.classes))
        else:
            model = resnet18(pretrained=False,
                                  num_classes=len(self.classes), shallow_linear=shallow)
        return model

    def remove_semantic_backdoors(self):
        """
        Semantic backdoors still occur with unmodified labels in the training
        set. This method removes them, so the only occurrence of the semantic
        backdoor will be in the
        :return: None
        """

        all_images = set(range(len(self.train_dataset)))
        unpoisoned_images = list(all_images.difference(set(
            self.params.poison_images)))

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       sampler=SubsetRandomSampler(
                                           unpoisoned_images))
    # add denormalize 2025/7/5 - zhongjie
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverses the normalization on a given tensor.

        Args:
            tensor (torch.Tensor): The normalized input tensor. Can be a single 
                                   image (C, H, W) or a batch (B, C, H, W).

        Returns:
            torch.Tensor: The denormalized tensor.
        """
        # Ensure mean and std are on the same device as the input tensor
        mean = torch.tensor(self.normalize.mean, device=tensor.device)
        std = torch.tensor(self.normalize.std, device=tensor.device)

        # Handle single image (3D) and batch of images (4D)
        if tensor.dim() == 3:  # For a single image (C, H, W)
            # Reshape to (C, 1, 1)
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif tensor.dim() == 4:  # For a batch of images (B, C, H, W)
            # Reshape to (1, C, 1, 1)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        
        # Apply the inverse transformation
        denormalized_tensor = tensor * std + mean
        
        return denormalized_tensor

