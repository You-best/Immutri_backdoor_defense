import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset 

from utils.utils import detect_malicious_threshold

class FeatureDatasets(Dataset):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        feature, label, is_bd = self.dataset[idx]
        return feature.to(self.device), label, is_bd

    def to_dict(self):
        return {
            "dataset" : self.dataset,
        }

def get_losses_and_data(model, dataloader, device):
    """
    Iterates through the dataset to calculate per-sample loss and collect all data.

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): The data loader.
        device (torch.device): The computation device (e.g., 'cuda:0').

    Returns:
        tuple: A tuple of four complete tensors containing losses, images, labels, and backdoor flags.
    """
    model.to(device)
    model.eval()

    all_losses, all_imgs, all_labels, all_is_bd = [], [], [], []

    with torch.no_grad():
        for imgs, labels, is_bd in tqdm(dataloader, desc="1/4: Calculating Losses"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            out = model(imgs)
            loss = F.cross_entropy(out, labels, reduction='none')
            
            all_losses.append(loss.cpu())
            all_imgs.append(imgs.cpu())
            all_labels.append(labels.cpu())
            all_is_bd.append(is_bd.cpu())

    # Concatenate lists of batch tensors into single large tensors
    losses_tensor = torch.cat(all_losses)
    imgs_tensor = torch.cat(all_imgs)
    labels_tensor = torch.cat(all_labels)
    is_bd_tensor = torch.cat(all_is_bd)
    
    return losses_tensor, imgs_tensor, labels_tensor, is_bd_tensor


def extract_features(model, imgs_tensor, device, batch_size=128):
    """
    Extracts features from a given tensor of images.
    Note: The original implementation processed images one-by-one, which is inefficient.
    This version has been optimized for batch processing.

    Args:
        model (nn.Module): The original model.
        imgs_tensor (torch.Tensor): A tensor containing all images.
        device (torch.device): The computation device.
        batch_size (int): The batch size to use for feature extraction.

    Returns:
        list: A list of feature tensors.
    """
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    feature_data = []
    
    # Use batch processing for efficiency
    num_samples = len(imgs_tensor)
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="2/4: Extracting Features"):
            batch_imgs = imgs_tensor[i:i+batch_size].to(device)
            
            # Remove the final layers (e.g., flatten and linear) to get feature maps
            features = feature_extractor(batch_imgs)
            # Global average pooling to get a fixed-size feature vector
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1) # (batch_size, num_features)
            
            feature_data.extend(features.cpu())
            
    return feature_data


def calculate_class_prototypes(feature_data, labels_tensor, is_bd_tensor, trust_indices, suspicion_indices):
    """
    Calculates the mean feature prototype for each class, separated by trusted and suspicious indices.

    Args:
        feature_data (list or torch.Tensor): A list or tensor containing features for all samples.
        labels_tensor (torch.Tensor): Labels for all samples.
        is_bd_tensor (torch.Tensor): Backdoor flags for all samples.
        trust_indices (list or set): Indices of trusted samples.
        suspicion_indices (list or set): Indices of suspicious samples.

    Returns:
        tuple: (rt_prototypes, rs_prototypes, rt_counts, rs_counts)
               RT_prototypes (dict): Prototypes for trusted classes {class_idx: feature_tensor}.
               RS_prototypes (dict): Prototypes for suspicious classes {class_idx: feature_tensor}.
               rt_counts (list): Sample counts for trusted classes.
               rs_counts (list): Sample counts for suspicious classes.
    """
    num_classes = labels_tensor.max().item() + 1
    
    # Helper function to compute prototypes
    def _compute_prototypes(indices):
        prototypes = {}
        counts = [0] * num_classes
        
        for idx in indices:
            label = labels_tensor[idx].item()
            feature = feature_data[idx]
            
            counts[label] += 1
            if label not in prototypes:
                prototypes[label] = feature.clone()
            else:
                prototypes[label] += feature
                
        for label, total_feature in prototypes.items():
            prototypes[label] = total_feature / counts[label]
            
        return dict(sorted(prototypes.items())), counts

    print("3/4: Calculating prototypes for Suspicion set...")
    rs_prototypes, rs_counts = _compute_prototypes(suspicion_indices)
    
    print("4/4: Calculating prototypes for Trust set...")
    rt_prototypes, rt_counts = _compute_prototypes(trust_indices)
    
    return rt_prototypes, rs_prototypes, rt_counts, rs_counts


def analyze_and_partition_dataset(model, dataloader, param):
    """
    Orchestrates the entire analysis and partitioning workflow, replacing the old NonVoting function.

    Args:
        model (nn.Module): The model to be analyzed.
        dataloader (DataLoader): The data loader with mixed data.
        param (object): A parameter object containing configuration like the device.

    Returns:
        dict: A dictionary containing all important results, accessible by key.
    """
    # Step 1: Calculate losses and get all data
    losses_tensor, imgs_tensor, labels_tensor, is_bd_tensor = get_losses_and_data(model, dataloader, param.device)

    # Step 2: Identify suspicious samples based on loss
    loss_threshold, malicious_indices = detect_malicious_threshold(losses_tensor.numpy(), plot=True)
    
    # Step 3: Extract features for all samples
    feature_data_list = extract_features(model, imgs_tensor, param.device)
    
    # Create the feature dataset object
    feature_dataset = FeatureDatasets(list(zip(feature_data_list, labels_tensor, is_bd_tensor)), device=param.device)

    # Step 4: Calculate class prototypes
    all_indices = set(range(len(labels_tensor)))
    suspicion_indices = set(malicious_indices)
    trust_indices = all_indices - suspicion_indices
    
    rt_prototypes, rs_prototypes, rt_counts, rs_counts = calculate_class_prototypes(
        feature_data_list, labels_tensor, is_bd_tensor, trust_indices, suspicion_indices
    )
    
    # --- START OF CORRECTION ---
    # Pack all results into a dictionary for clarity and ease of access
    # We pass the prototype DICTIONARIES directly, not lists of their values.
    results = {
        "rt_prototypes": rt_prototypes, # MODIFIED: Pass the dictionary
        "rs_prototypes": rs_prototypes, # MODIFIED: Pass the dictionary
        "rt_counts": rt_counts,
        "rs_counts": rs_counts,
        "feature_dataset": feature_dataset,
        "loss_threshold": loss_threshold,
        "all_losses": losses_tensor,
        "malicious_indices": malicious_indices,
        "suspicion_indices": suspicion_indices,
        "trust_indices": list(trust_indices),
        "all_imgs": imgs_tensor,
        "all_labels": labels_tensor,
        "all_is_bd": is_bd_tensor,
    }
    # --- END OF CORRECTION ---
    
    return results



