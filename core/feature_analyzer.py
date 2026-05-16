import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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


def detect_first_minimum_loss(losses_tensor, plot=False):
    """
    Detect the first minimum loss operating point from the loss spectrum.
    This is a conservative threshold selection strategy for LGFD.
    
    Args:
        losses_tensor (torch.Tensor): Per-sample losses
        plot (bool): Whether to plot the density curve
        
    Returns:
        tuple: (threshold, indices_below_threshold)
    """
    individual_losses_np = losses_tensor.numpy()
    sorted_losses = np.sort(individual_losses_np)
    
    # Use KDE to find the first minimum after the first peak
    bandwidth = 0.05
    individual_losses_reshaped = individual_losses_np.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(individual_losses_reshaped)
    log_density_sorted = kde.score_samples(sorted_losses.reshape(-1, 1))
    density_values = np.exp(log_density_sorted)
    
    if plot:
        plt.figure(figsize=(5, 3))
        plt.plot(sorted_losses, density_values, 'b-', label='Estimated Density')
        plt.fill_between(sorted_losses, density_values, alpha=0.5)
        plt.xlabel('Loss Value')
        plt.ylabel('Density')
        plt.title('Loss Distribution with First Minimum Detection')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Find peaks and valleys
    peaks, _ = find_peaks(density_values)
    valleys, _ = find_peaks(-density_values)
    
    # Find the first peak
    if len(peaks) == 0:
        final_threshold = 0.1
    else:
        first_peak_index = peaks[0]
        
        # Find the first valley after the first peak
        threshold_index = None
        for valley in valleys:
            if valley > first_peak_index:
                threshold_index = valley
                break
        
        if threshold_index is None:
            final_threshold = sorted_losses[int(len(sorted_losses) * 0.1)]
        else:
            final_threshold = sorted_losses[threshold_index]
    
    malicious_indices = np.where(individual_losses_np <= final_threshold)[0]
    return final_threshold, malicious_indices


def infer_target_label(model, dataloader, suspicious_indices, device, num_classes=10):
    """
    Infer the likely target label by analyzing prediction patterns on suspicious samples.
    
    Args:
        model: The backdoor model
        dataloader: Data loader
        suspicious_indices: Indices of suspicious samples
        device: Computation device
        num_classes: Number of classes
        
    Returns:
        int: Inferred target label
    """
    model.eval()
    label_counts = torch.zeros(num_classes)
    
    suspicious_set = set(suspicious_indices.tolist() if isinstance(suspicious_indices, np.ndarray) else suspicious_indices)
    
    with torch.no_grad():
        global_idx = 0
        for imgs, labels, is_bd in tqdm(dataloader, desc="Inferring target label"):
            batch_size = len(imgs)
            batch_indices = range(global_idx, global_idx + batch_size)
            
            # Find suspicious samples in this batch
            batch_suspicious_mask = torch.tensor([i in suspicious_set for i in batch_indices])
            
            if batch_suspicious_mask.sum() > 0:
                suspicious_imgs = imgs[batch_suspicious_mask].to(device)
                outputs = model(suspicious_imgs)
                predictions = torch.argmax(outputs, dim=1)
                
                for pred in predictions:
                    label_counts[pred.item()] += 1
            
            global_idx += batch_size
    
    inferred_target = torch.argmax(label_counts).item()
    print(f"Inferred target label: {inferred_target} (confidence: {label_counts[inferred_target].item() / max(label_counts.sum(), 1):.2%})")
    return inferred_target


def directional_feature_matching(feature_data, labels_tensor, inferred_target, trust_indices, suspicion_indices, param):
    """
    Refine the candidate set through class-conditional feature matching.
    Uses directional matching to identify samples that align with the backdoor direction.
    
    Args:
        feature_data: List of feature tensors
        labels_tensor: Labels for all samples
        inferred_target: The inferred target label
        trust_indices: Indices of trusted samples
        suspicion_indices: Indices of suspicious samples
        param: Parameter object
        
    Returns:
        dict: Refined partition results
    """
    num_classes = labels_tensor.max().item() + 1
    
    # Calculate prototypes for the inferred target class
    target_trust_features = []
    target_suspicion_features = []
    
    for idx in trust_indices:
        if labels_tensor[idx].item() == inferred_target:
            target_trust_features.append(feature_data[idx])
    
    for idx in suspicion_indices:
        if labels_tensor[idx].item() == inferred_target:
            target_suspicion_features.append(feature_data[idx])
    
    if len(target_trust_features) == 0 or len(target_suspicion_features) == 0:
        print("Warning: Insufficient features for target class refinement")
        return {
            "refined_suspicion": suspicion_indices,
            "refined_trust": trust_indices,
            "target_prototype_trust": None,
            "target_prototype_suspicion": None
        }
    
    # Compute mean prototypes
    target_trust_proto = torch.stack(target_trust_features).mean(dim=0)
    target_suspicion_proto = torch.stack(target_suspicion_features).mean(dim=0)
    
    # Calculate directional vector (backdoor direction)
    backdoor_direction = target_suspicion_proto - target_trust_proto
    backdoor_direction_norm = backdoor_direction / (backdoor_direction.norm() + 1e-12)
    
    # Project all target-class samples onto the backdoor direction
    refined_suspicion = []
    refined_trust = list(trust_indices)
    
    for idx in suspicion_indices:
        if labels_tensor[idx].item() == inferred_target:
            feature = feature_data[idx]
            # Project onto backdoor direction
            projection = torch.dot(feature - target_trust_proto, backdoor_direction_norm)
            
            # If projection is positive and significant, it's likely poisoned
            if projection > 0:
                refined_suspicion.append(idx)
            else:
                refined_trust.append(idx)
    
    print(f"Refined suspicion set size: {len(refined_suspicion)}")
    print(f"Refined trust set size: {len(refined_trust)}")
    
    return {
        "refined_suspicion": refined_suspicion,
        "refined_trust": refined_trust,
        "target_prototype_trust": target_trust_proto,
        "target_prototype_suspicion": target_suspicion_proto,
        "backdoor_direction": backdoor_direction
    }


def analyze_and_partition_dataset(model, dataloader, param):
    """
    LGFD Stage: Loss-Guided Feature Decoupling
    Orchestrates the complete analysis with target label inference and feature refinement.
    
    Args:
        model (nn.Module): The delivered backdoor model
        dataloader (DataLoader): Mixed training data loader
        param (object): Parameter object with configuration
        
    Returns:
        dict: Complete analysis results including refined partitions
    """
    print("\n=== LGFD Stage: Loss-Guided Feature Decoupling ===")
    
    # Step 1: Calculate losses and collect all data
    print("Step 1: Computing per-sample losses...")
    losses_tensor, imgs_tensor, labels_tensor, is_bd_tensor = get_losses_and_data(model, dataloader, param.device)
    
    # Step 2: Detect first minimum loss operating point (conservative threshold)
    print("Step 2: Detecting first minimum loss operating point...")
    loss_threshold, initial_suspicious_indices = detect_first_minimum_loss(losses_tensor, plot=True)
    print(f"Initial suspicious samples: {len(initial_suspicious_indices)} / {len(losses_tensor)}")
    
    # Step 3: Extract features for all samples
    print("Step 3: Extracting penultimate-layer features...")
    feature_data_list = extract_features(model, imgs_tensor, param.device)
    
    # Step 4: Infer the likely target label
    print("Step 4: Inferring attack target label...")
    inferred_target = infer_target_label(
        model, dataloader, initial_suspicious_indices, param.device, param.num_classes
    )
    
    # Step 5: Directional feature matching for refinement
    print("Step 5: Performing directional feature matching...")
    all_indices = set(range(len(labels_tensor)))
    initial_suspicion_set = set(
        initial_suspicious_indices.tolist() if isinstance(initial_suspicious_indices, np.ndarray) 
        else initial_suspicious_indices
    )
    initial_trust_indices = all_indices - initial_suspicion_set
    
    refinement_results = directional_feature_matching(
        feature_data_list, labels_tensor, inferred_target,
        initial_trust_indices, initial_suspicion_set, param
    )
    
    # Calculate class prototypes for both sets
    rt_prototypes, rs_prototypes, rt_counts, rs_counts = calculate_class_prototypes(
        feature_data_list, labels_tensor, is_bd_tensor,
        refinement_results["refined_trust"], refinement_results["refined_suspicion"]
    )
    
    # Create feature dataset
    feature_dataset = FeatureDatasets(
        list(zip(feature_data_list, labels_tensor, is_bd_tensor)), 
        device=param.device
    )
    
    # Pack all results
    results = {
        "rt_prototypes": rt_prototypes,
        "rs_prototypes": rs_prototypes,
        "rt_counts": rt_counts,
        "rs_counts": rs_counts,
        "feature_dataset": feature_dataset,
        "loss_threshold": loss_threshold,
        "all_losses": losses_tensor,
        "all_imgs": imgs_tensor,
        "all_labels": labels_tensor,
        "all_is_bd": is_bd_tensor,
        "inferred_target": inferred_target,
        "refined_suspicion_indices": refinement_results["refined_suspicion"],
        "refined_trust_indices": refinement_results["refined_trust"],
        "target_prototype_trust": refinement_results["target_prototype_trust"],
        "target_prototype_suspicion": refinement_results["target_prototype_suspicion"],
        "backdoor_direction": refinement_results.get("backdoor_direction", None)
    }
    
    print("✅ LGFD Stage completed successfully.")
    return results



