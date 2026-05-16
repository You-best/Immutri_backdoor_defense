import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.resnet_ import resnet18, resnet101
from tasks.task import Task
import torch

class Imagenet10Task(Task):

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverses the normalization on a tensor image.
        
        The synthesizer needs this to apply triggers to the raw pixel values.
        
        Args:
            tensor (torch.Tensor): A normalized tensor image of shape 
                                   (C, H, W) or (B, C, H, W).
        
        Returns:
            torch.Tensor: The denormalized tensor image in the [0, 1] range.
        """
        # The logic is: denormalized = (normalized * std) + mean
        # We need to reshape mean and std to broadcast correctly across the tensor.
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        # The normalization transform to be used in data loading
        self.normalize = transforms.Normalize(self.mean, self.std)
        
        mean = self.mean.to(tensor.device).view(1, -1, 1, 1)
        std = self.std.to(tensor.device).view(1, -1, 1, 1)

        # Check if the input is a single image or a batch of images
        was_single_image = len(tensor.shape) == 3
        if was_single_image:
            tensor = tensor.unsqueeze(0)

        # Apply the denormalization
        denormalized_tensor = tensor * std + mean

        # If it was a single image, remove the batch dimension we added
        if was_single_image:
            denormalized_tensor = denormalized_tensor.squeeze(0)

        return denormalized_tensor

    def load_data(self):

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.train_dataset = torchvision.datasets.ImageNet(
            root=self.params.data_path,
            split='train', transform=train_transform)

        self.test_dataset = torchvision.datasets.ImageNet(
            root=self.params.data_path,
            split='val', transform=test_transform)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True, num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=8, pin_memory=True)

        self.classes = ['hammerhead shark', 'eagle', 'giant schnauzer', 'ballpen', 'cleaver',
                       'holster', 'library', 'pickup', 'studio couch', 'wing']

    def build_model(self) -> None:
        return resnet18(pretrained=self.params.pretrained, num_classes=10)
 