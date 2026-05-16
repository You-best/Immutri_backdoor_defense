# Filename: data_purification.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from utils.utils import CustomDataset

from utils.utils import get_cosine_similarity_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def purify_and_generate_candidates(analysis_results, target_label, param):
    """
    Purifies the suspicious prototype for a specific target class and generates a series of candidate prototypes.

    Args:
        analysis_results (dict): The results dictionary from the analyze_and_partition_dataset function.
        target_label (int): The target class label you want to analyze.

    Returns:
        tuple: (candidate_prototypes, purified_prototype_b, cosine_similarity_matrix)
    """
    # --- 1. Dynamically find the index and data for the target class ---
    
    # Get prototype dictionaries and count list from the results
    # Now this will correctly receive dictionaries as intended.
    rs_prototypes_dict = analysis_results["rs_prototypes"]
    rt_prototypes_dict = analysis_results["rt_prototypes"]

    # --- START OF CORRECTION ---
    # Corrected the key to get the suspicious counts
    rs_counts = analysis_results["rs_counts"] # MODIFIED: was "rt_counts"
    # --- END OF CORRECTION ---
    
    feature_dataset = analysis_results["feature_dataset"]

    # This check will now work correctly because rs_prototypes_dict is a dictionary
    if target_label not in rs_prototypes_dict:
        raise KeyError(f"Target label {target_label} not found in the set of suspicious prototypes. "
                       f"Available labels are: {list(rs_prototypes_dict.keys())}")
    if target_label not in rt_prototypes_dict:
        raise KeyError(f"Target label {target_label} not found in the set of trusted prototypes. "
                       f"Available labels are: {list(rt_prototypes_dict.keys())}")


    # Get prototypes and count by label
    target_trust_prototype = rt_prototypes_dict[target_label].flatten()
    target_suspicion_prototype = rs_prototypes_dict[target_label].flatten()
    target_suspicion_count = rs_counts[target_label]

    # ... (The rest of the function remains the same) ...
    print(f"Successfully found data for target label: {target_label}")

    # --- 2. Calculate cosine similarity and projection ---
    dot_product = torch.dot(target_trust_prototype, target_suspicion_prototype)
    magnitude_trust_prototype = target_trust_prototype.norm()
    magnitude_suspicion_prototype = target_suspicion_prototype.norm()
    cosine_similarity = dot_product / (magnitude_trust_prototype * magnitude_suspicion_prototype)
    
    print(f"Magnitude of Trust Prototype: {magnitude_trust_prototype:.4f}")
    print(f"Magnitude of Suspicion Prototype: {magnitude_suspicion_prototype:.4f}")
    print(f"Cosine Similarity between prototypes: {cosine_similarity:.4f}")

    # --- 3. Calculate the purified prototype b ---
    projection_scalar_m = (target_suspicion_count * magnitude_suspicion_prototype * cosine_similarity) / magnitude_trust_prototype
    purified_prototype_b = (target_suspicion_count * target_suspicion_prototype - projection_scalar_m * target_trust_prototype) / (target_suspicion_count - projection_scalar_m)
    
    print(f"Suspicion Count for target: {target_suspicion_count}")
    print(f"Projection Scalar (m): {projection_scalar_m:.4f}")
    print(f"Shape of Purified Prototype (b): {purified_prototype_b.shape}")

    # --- 4. Generate a list of candidate prototypes ---
    candidate_prototypes = []
    num_candidates = 2000
    shift_factor = 0.1
    for i in range(num_candidates):
        shifted_prototype = purified_prototype_b + shift_factor * i * target_trust_prototype
        candidate_prototypes.append(shifted_prototype)

    print(f"Generated {len(candidate_prototypes)} candidate prototypes.")

    # --- 5. Calculate the final similarity matrix ---
    cosine_similarity_matrix = get_cosine_similarity_matrix(feature_dataset, candidate_prototypes, param)
    
    return candidate_prototypes, purified_prototype_b, cosine_similarity_matrix


def identify_suspicious_by_similarity(
    cosine_similarity_matrix,
    similarity_threshold=0.56,
    candidate_index_threshold=300
):
    """
    Identifies an initial set of suspicious samples based on their similarity to candidate prototypes.

    Args:
        cosine_similarity_matrix (torch.Tensor): A matrix of shape [num_samples, num_candidates].
        similarity_threshold (float): The similarity score above which a sample is considered suspicious.
        candidate_index_threshold (int): If a sample's best match is a candidate with an index
                                         below this value, it is considered clean (not suspicious).

    Returns:
        list: A list of indices for samples flagged as suspicious.
    """
    num_samples = cosine_similarity_matrix.shape[0]
    is_suspicious_mask = [False] * num_samples

    for i in tqdm(range(num_samples), desc="Filtering by similarity"):
        max_similarity, max_index = torch.max(cosine_similarity_matrix[i], dim=0)
        
        # Rule 1: High similarity suggests a suspicious sample
        if max_similarity > similarity_threshold:
            is_suspicious_mask[i] = True
        
        # Rule 2: If the best-matching candidate is "early" (i.e., less shifted),
        # it's likely a clean sample misidentified. Override the suspicion.
        if max_index.item() < candidate_index_threshold:
            is_suspicious_mask[i] = False
            
    suspicious_indices = [idx for idx, is_suspicious in enumerate(is_suspicious_mask) if is_suspicious]
    return suspicious_indices


def partition_sets_by_loss(
    all_losses,
    all_labels,
    all_is_bd,
    initial_suspicious_indices,
    inspection_set_ratio=0.1,
    clean_set_ratio=0.5
):
    """
    Partitions the dataset into a final inspection set and a clean set based on loss.

    Args:
        all_losses (torch.Tensor): Losses for all samples.
        all_labels (torch.Tensor): Labels for all samples.
        all_is_bd (torch.Tensor): Backdoor flags for all samples.
        initial_suspicious_indices (list): Indices from the similarity filtering step.
        inspection_set_ratio (float): The fraction of lowest-loss samples to keep from the suspicious set.
        clean_set_ratio (float): The fraction of highest-loss samples to keep from the trusted set.

    Returns:
        tuple: (final_inspection_indices, final_clean_indices)
    """
    print("Step 3/5: Partitioning data into final inspection and clean sets...")
    
    # --- Create the Inspection Set (low-loss from suspicious samples) ---
    suspicion_losses = all_losses[initial_suspicious_indices]
    sorted_loss_indices = torch.argsort(suspicion_losses, descending=False)
    
    top_n_inspection = int(len(suspicion_losses) * inspection_set_ratio)
    final_inspection_indices = [initial_suspicious_indices[i] for i in sorted_loss_indices[:top_n_inspection]]

    # --- Create the Clean Set (high-loss from trusted samples) ---
    num_total_samples = len(all_losses)
    initial_suspicious_set = set(initial_suspicious_indices)
    trusted_indices = [i for i in range(num_total_samples) if i not in initial_suspicious_set]
    
    trusted_losses = all_losses[trusted_indices]
    sorted_loss_indices_clean = torch.argsort(trusted_losses, descending=True)
    
    top_n_clean = int(len(trusted_losses) * clean_set_ratio)
    final_clean_indices = [trusted_indices[i] for i in sorted_loss_indices_clean[:top_n_clean]]
    
    # --- Print statistics for verification ---
    print("\n--- Data Partitioning Statistics ---")
    s_count = len(initial_suspicious_indices)
    bd_in_s_count = sum(1 for i in initial_suspicious_indices if all_is_bd[i] == 1)
    print(f"Initial suspicious samples identified: {s_count}")
    if s_count > 0:
        print(f"Backdoor samples within initial suspicious set: {bd_in_s_count} ({bd_in_s_count / s_count:.2%})")

    bd_in_inspection_count = sum(1 for i in final_inspection_indices if all_is_bd[i] == 1)
    print(f"\nFinal Inspection Set (lowest {inspection_set_ratio:.0%} loss from suspicious): {len(final_inspection_indices)} samples")
    if len(final_inspection_indices) > 0:
        print(f"Backdoor samples within final inspection set: {bd_in_inspection_count} ({bd_in_inspection_count / len(final_inspection_indices):.2%})")

    bd_in_clean_count = sum(1 for i in final_clean_indices if all_is_bd[i] == 1)
    print(f"\nFinal Clean Set (highest {clean_set_ratio:.0%} loss from trusted): {len(final_clean_indices)} samples")
    if len(final_clean_indices) > 0:
        print(f"Backdoor samples remaining in final clean set: {bd_in_clean_count} ({bd_in_clean_count/len(final_clean_indices):.2%})")
    print("------------------------------------")
    
    return final_inspection_indices, final_clean_indices


def create_partitioned_dataloaders(
    all_imgs,
    all_labels,
    inspection_indices,
    clean_indices,
    batch_size=64
):
    """
    Creates Dataset and DataLoader objects for the final partitioned sets.

    Args:
        all_imgs (torch.Tensor): Tensor of all images.
        all_labels (torch.Tensor): Tensor of all labels.
        inspection_indices (list): List of indices for the inspection set.
        clean_indices (list): List of indices for the clean set.
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        tuple: (inspection_set_dataloader, clean_set_dataloader)
    """
    print("Step 4/5: Creating final DataLoaders...")
    
    # Create Inspection Set
    inspection_images = all_imgs[inspection_indices]
    inspection_labels = all_labels[inspection_indices]
    inspection_dataset = CustomDataset(images=inspection_images, labels=inspection_labels)
    inspection_dataloader = DataLoader(inspection_dataset, batch_size=batch_size, shuffle=True)
    
    # Create Clean Set
    clean_images = all_imgs[clean_indices]
    clean_labels = all_labels[clean_indices]
    clean_dataset = CustomDataset(images=clean_images, labels=clean_labels)
    clean_dataloader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Inspection Set DataLoader created with {len(inspection_dataset)} samples.")
    print(f"Clean Set DataLoader created with {len(clean_dataset)} samples.")
    
    return inspection_dataloader, clean_dataloader


# --- Main Orchestrator Function ---

def create_purified_partitions(analysis_results, param, config):
    """
    Main orchestrator function that runs the full purification and partitioning pipeline.

    Args:
        analysis_results (dict): The dictionary from the initial analysis step.
        param (object): Your parameter object, containing `param.target`.
        config (dict): A dictionary for hyperparameters like thresholds and ratios.

    Returns:
        tuple: (inspection_dataloader, clean_dataloader)
    """
    print("\n--- Starting Data Purification and Partitioning Pipeline ---")
    
    # Step 1: Purify prototype and generate candidates to get the similarity matrix
    # cosine_similarity_matrix = purify_and_generate_candidates(analysis_results, param.target, param)
    _, _, cosine_similarity_matrix = purify_and_generate_candidates(
    analysis_results, param.target, param
    )

    # Step 2: Identify an initial set of suspicious samples
    initial_suspicious_indices = identify_suspicious_by_similarity(
        cosine_similarity_matrix,
        similarity_threshold=config['similarity_threshold'],
        candidate_index_threshold=config['candidate_index_threshold']
    )

    # Step 3: Partition the dataset into final inspection and clean sets based on loss
    final_inspection_indices, final_clean_indices = partition_sets_by_loss(
        all_losses=analysis_results['all_losses'],
        all_labels=analysis_results['all_labels'],
        all_is_bd=analysis_results['all_is_bd'],
        initial_suspicious_indices=initial_suspicious_indices,
        inspection_set_ratio=config['inspection_set_ratio'],
        clean_set_ratio=config['clean_set_ratio']
    )

    # Step 4: Create the final DataLoaders
    inspection_dataloader, clean_dataloader = create_partitioned_dataloaders(
        all_imgs=analysis_results['all_imgs'],
        all_labels=analysis_results['all_labels'],
        inspection_indices=final_inspection_indices,
        clean_indices=final_clean_indices,
        batch_size=config['batch_size']
    )
    
    print("\n✅ Pipeline Finished Successfully.")
    return inspection_dataloader, clean_dataloader


def get_filtered_data(model_filter, mixed_dataloader, loss_threshold, param):
    model_filter = model_filter.to(param.device)  # Use model_filter instead of model_test
    model_filter.eval()  # Switch to evaluation mode
    
    # Define loss function using cross entropy, with 'none' reduction to compute per-sample loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Store filtered sample indices
    filtered_idxs = []
    
    # Variables to calculate the metrics
    is_bd_list = []  # List to track whether samples are poisoned (ground truth)
    bool_list = []  # List to track if samples are detected as poisoned by the model
    correct_poisoned = 0  # Counter for correctly detected poisoned samples
    misclassified_clean_as_poisoned = 0  # Counter for clean samples misclassified as poisoned

    # Metrics for classification (TP, FP, TN, FN)
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # Initialize global index to track global positions of samples
    global_idx = 0 
    
    for data in tqdm(mixed_dataloader, desc="Filtering Data"):
        inputs, labels, is_bd = data  # Data includes inputs, labels, and ground truth poisoned labels
        inputs = inputs.to(param.device)
        labels = labels.to(param.device)
        
        is_bd_list.extend(is_bd.cpu().numpy())  # Collect ground truth poisoned labels
        
        with torch.no_grad():
            # Forward pass: compute model outputs
            outputs = model_filter(inputs)  # Use model_filter for inference
            
            # Compute loss for each sample
            loss = criterion(outputs, labels)
        
        # Get the sample losses as a NumPy array
        sample_losses = loss.detach().cpu().numpy()
        
        # Create a mask where loss is greater than the threshold (clean samples)
        mask = sample_losses >= loss_threshold  # Mark clean samples with high loss
        
        # The poisoned samples will be the ones where loss is less than the threshold
        poisoned_mask = ~mask  # Use the inverse mask for poisoned samples
        
        # Track which samples are detected as poisoned by the model
        bool_list.extend(poisoned_mask)  # Append poisoned mask values for this batch
        
        # Count correctly identified poisoned samples
        correct_poisoned += sum([1 for i in range(len(poisoned_mask)) if poisoned_mask[i] and is_bd[i]])  # Detected and actually poisoned
        
        # Count clean samples misclassified as poisoned
        misclassified_clean_as_poisoned += sum([1 for i in range(len(poisoned_mask)) if poisoned_mask[i] and not is_bd[i]])  # Clean but detected as poisoned
        
        # Count TP, FP, TN, FN
        for i in range(len(poisoned_mask)):
            if poisoned_mask[i] and is_bd[i]:
                tp += 1  # True Positive
            elif poisoned_mask[i] and not is_bd[i]:
                fp += 1  # False Positive
            elif not poisoned_mask[i] and not is_bd[i]:
                tn += 1  # True Negative
            elif not poisoned_mask[i] and is_bd[i]:
                fn += 1  # False Negative
        
        # Get global indices of the filtered samples (those with loss < threshold)
        batch_size = len(inputs)
        batch_filtered_idxs = [global_idx + i for i, valid in enumerate(mask) if valid]
        
        # Add filtered indices to the list
        filtered_idxs.extend(batch_filtered_idxs)
        
        # Update global index
        global_idx += batch_size

    # Calculate Precision, Recall, Accuracy, F1 Score, TPR, FPR
    precision = precision_score(is_bd_list, bool_list)
    recall = recall_score(is_bd_list, bool_list)
    accuracy = accuracy_score(is_bd_list, bool_list)
    f1 = f1_score(is_bd_list, bool_list)
    
    # True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0

    # Output the metrics
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
    # Return filtered indices
    print(f"Number of filtered samples: {len(filtered_idxs)}")
    return filtered_idxs


