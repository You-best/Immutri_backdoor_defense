# Filename: data_purification.py
# TMFS (Tri-Model Functional Synergy) Implementation for IMMUTRI

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from utils.utils import CustomDataset
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import copy


def train_auxiliary_detector_with_uniform_noise(model, dataloader, param, num_epochs=5, lr=0.001, noise_rate=0.3):
    """
    TMFS Stage: Train an auxiliary detector by fine-tuning with fresh uniform label noise.
    
    Args:
        model: Copy of the delivered model to be fine-tuned
        dataloader: Training data loader
        param: Parameter object
        num_epochs: Number of training epochs
        lr: Learning rate
        noise_rate: Fraction of samples to receive uniform random labels
        
    Returns:
        nn.Module: Trained auxiliary detector (fenh)
    """
    print("\n=== TMFS Stage: Training Auxiliary Detector ===")
    fenh = copy.deepcopy(model)
    fenh.train()
    
    optimizer = optim.Adam(fenh.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_classes = param.num_classes
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for imgs, labels, is_bd in progress_bar:
            imgs = imgs.to(param.device)
            labels = labels.to(param.device)
            batch_size = len(imgs)
            
            # Apply uniform label noise
            noisy_labels = labels.clone()
            noise_mask = torch.rand(batch_size, device=param.device) < noise_rate
            if noise_mask.sum() > 0:
                noisy_labels[noise_mask] = torch.randint(0, num_classes, (noise_mask.sum(),), device=param.device)
            
            # Forward pass
            outputs = fenh(imgs)
            loss = criterion(outputs, noisy_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch_size
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()  # Track against true labels
            total += batch_size
        
        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Clean Accuracy={accuracy:.4f}")
    
    print("✅ Auxiliary detector trained successfully.")
    return fenh


def purify_dataset_with_detector(fenh, dataloader, param, loss_threshold=None):
    """
    Use the auxiliary detector to purify the training set.
    Samples with high loss are considered clean and retained.
    
    Args:
        fenh: Auxiliary detector model
        dataloader: Full training data loader
        param: Parameter object
        loss_threshold: Optional threshold; if None, uses adaptive thresholding
        
    Returns:
        tuple: (retained_indices, purified_dataloader)
    """
    print("\n=== Dataset Purification ===")
    fenh.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    all_losses = []
    all_imgs = []
    all_labels = []
    all_is_bd = []
    
    with torch.no_grad():
        for imgs, labels, is_bd in tqdm(dataloader, desc="Computing losses for purification"):
            imgs = imgs.to(param.device)
            labels = labels.to(param.device)
            
            outputs = fenh(imgs)
            losses = criterion(outputs, labels)
            
            all_losses.append(losses.cpu())
            all_imgs.append(imgs.cpu())
            all_labels.append(labels.cpu())
            all_is_bd.append(is_bd.cpu())
    
    losses_tensor = torch.cat(all_losses)
    imgs_tensor = torch.cat(all_imgs)
    labels_tensor = torch.cat(all_labels)
    is_bd_tensor = torch.cat(all_is_bd)
    
    # Determine threshold
    if loss_threshold is None:
        # Use median loss as threshold
        loss_threshold = losses_tensor.median().item()
        print(f"Using median loss as threshold: {loss_threshold:.4f}")
    
    # Retain samples with loss >= threshold (likely clean)
    retained_mask = losses_tensor >= loss_threshold
    retained_indices = torch.where(retained_mask)[0].tolist()
    
    # Create purified dataset
    purified_imgs = imgs_tensor[retained_mask]
    purified_labels = labels_tensor[retained_mask]
    purified_dataset = CustomDataset(images=purified_imgs, labels=purified_labels)
    purified_dataloader = DataLoader(purified_dataset, batch_size=param.batch_size, shuffle=True)
    
    # Statistics
    total_samples = len(losses_tensor)
    retained_count = len(retained_indices)
    poisoned_in_retained = is_bd_tensor[retained_mask].sum().item()
    
    print(f"Total samples: {total_samples}")
    print(f"Retained samples: {retained_count} ({retained_count/total_samples:.2%})")
    print(f"Poisoned samples in retained set: {poisoned_in_retained} ({poisoned_in_retained/max(retained_count,1):.2%})")
    
    return retained_indices, purified_dataloader


def train_reference_model(task, purified_dataloader, param, num_epochs=10, lr=0.001):
    """
    Train a reference model (fref) on the purified dataset.
    
    Args:
        task: Task object to build model
        purified_dataloader: Purified training data loader
        param: Parameter object
        num_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        nn.Module: Trained reference model (fref)
    """
    print("\n=== Training Reference Model ===")
    fref = task.build_model().to(param.device)
    fref.train()
    
    optimizer = optim.Adam(fref.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(purified_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for imgs, labels in progress_bar:
            imgs = imgs.to(param.device)
            labels = labels.to(param.device)
            batch_size = len(imgs)
            
            outputs = fref(imgs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size
        
        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
    
    print("✅ Reference model trained successfully.")
    return fref


def tri_model_firewall_inference(fbd, fenh, fref, input_tensor, param, threshold=0.5):
    """
    Tri-Model Firewall for online inference.
    Uses three models to detect and reject backdoor inputs.
    
    Args:
        fbd: Original delivered model
        fenh: Auxiliary detector (trained with uniform noise)
        fref: Reference model (trained on purified data)
        input_tensor: Input sample(s)
        param: Parameter object
        threshold: Disagreement threshold for rejection
        
    Returns:
        tuple: (prediction, is_rejected)
    """
    fbd.eval()
    fenh.eval()
    fref.eval()
    
    with torch.no_grad():
        # Get predictions from all three models
        out_bd = fbd(input_tensor)
        out_enh = fenh(input_tensor)
        out_ref = fref(input_tensor)
        
        pred_bd = torch.argmax(out_bd, dim=1)
        pred_enh = torch.argmax(out_enh, dim=1)
        pred_ref = torch.argmax(out_ref, dim=1)
        
        # Check for disagreement
        agreement_count = (
            (pred_bd == pred_enh).float() +
            (pred_bd == pred_ref).float() +
            (pred_enh == pred_ref).float()
        )
        
        # If less than 2 agreements, reject the input
        is_rejected = agreement_count < 2
        
        # Use reference model's prediction if not rejected
        final_prediction = pred_ref.clone()
        final_prediction[is_rejected] = -1  # Mark as rejected
        
    return final_prediction, is_rejected


def get_filtered_data(model_filter, mixed_dataloader, loss_threshold, param):
    """
    Legacy function for filtering data based on loss threshold.
    Kept for backward compatibility.
    """
    model_filter = model_filter.to(param.device)
    model_filter.eval()
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    filtered_idxs = []
    
    is_bd_list = []
    bool_list = []
    correct_poisoned = 0
    misclassified_clean_as_poisoned = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    global_idx = 0 
    
    for data in tqdm(mixed_dataloader, desc="Filtering Data"):
        inputs, labels, is_bd = data
        inputs = inputs.to(param.device)
        labels = labels.to(param.device)
        
        is_bd_list.extend(is_bd.cpu().numpy())
        
        with torch.no_grad():
            outputs = model_filter(inputs)
            loss = criterion(outputs, labels)
        
        sample_losses = loss.detach().cpu().numpy()
        mask = sample_losses >= loss_threshold
        poisoned_mask = ~mask
        
        bool_list.extend(poisoned_mask)
        correct_poisoned += sum([1 for i in range(len(poisoned_mask)) if poisoned_mask[i] and is_bd[i]])
        misclassified_clean_as_poisoned += sum([1 for i in range(len(poisoned_mask)) if poisoned_mask[i] and not is_bd[i]])
        
        for i in range(len(poisoned_mask)):
            if poisoned_mask[i] and is_bd[i]:
                tp += 1
            elif poisoned_mask[i] and not is_bd[i]:
                fp += 1
            elif not poisoned_mask[i] and not is_bd[i]:
                tn += 1
            elif not poisoned_mask[i] and is_bd[i]:
                fn += 1
        
        batch_size = len(inputs)
        batch_filtered_idxs = [global_idx + i for i, valid in enumerate(mask) if valid]
        filtered_idxs.extend(batch_filtered_idxs)
        global_idx += batch_size

    precision = precision_score(is_bd_list, bool_list) if len(set(is_bd_list)) > 1 else 0
    recall = recall_score(is_bd_list, bool_list) if len(set(is_bd_list)) > 1 else 0
    accuracy = accuracy_score(is_bd_list, bool_list)
    f1 = f1_score(is_bd_list, bool_list) if len(set(is_bd_list)) > 1 else 0
    
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0

    print(f"Total samples: {len(is_bd_list)}")
    print(f"Poisoned samples (ground truth): {sum(is_bd_list)}")
    print(f"Detected poisoned samples (by model): {sum(bool_list)}")
    print(f"Correctly identified poisoned samples: {correct_poisoned}")
    print(f"Misclassified clean samples as poisoned: {misclassified_clean_as_poisoned}")    
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"True Negative Rate (TNR): {tnr:.4f}")    
    print(f"Number of filtered samples: {len(filtered_idxs)}")
    return filtered_idxs
