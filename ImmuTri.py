from config import parser_handle
import os
parser = parser_handle()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(3) 

from utils.parameters import My_Params, set_param

param = My_Params()
set_param(param, args)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

torch.manual_seed(123) 
torch.cuda.manual_seed(123) 
import copy

from utils.initial import get_task, applied_attack, get_dataloader
from utils.utils import evaluate, evaluate_model, CustomDataset
from core.feature_analyzer import analyze_and_partition_dataset
from core.data_purification import (
    train_auxiliary_detector_with_uniform_noise,
    purify_dataset_with_detector,
    train_reference_model,
    tri_model_firewall_inference
)
from torch.utils.data import DataLoader


def main():
    print("\n" + "="*80)
    print("IMMUTRI: Two-Stage Post-Training White-Box Defense")
    print("="*80)
    
    task = get_task(param)

    # Get the attack synthesizer
    synthesizer = applied_attack(param, task)
    
    # Get dataloaders
    clean_testset_loader, backdoor_testset_loader, mixed_dataloader = get_dataloader(
        param, task, synthesizer
    )

    # Load the delivered backdoor model (fbd)
    backdoor_model = task.build_model().to(param.device)
    save_dir = './save'
    os.makedirs(save_dir, exist_ok=True)
    backdoor_model_name = f"{param.dataset}_{param.attack}_backdoor_model.pth"
    backdoor_model_save_path = os.path.join(save_dir, backdoor_model_name)
    
    # Try to load pre-trained backdoor model
    if os.path.exists(backdoor_model_save_path):
        backdoor_model.load_state_dict(torch.load(backdoor_model_save_path))
        print(f"Loaded backdoor model from: {backdoor_model_save_path}")
    else:
        print(f"Warning: Backdoor model not found at {backdoor_model_save_path}")
        print("Please run attack.py first to generate the backdoor model.")
        return
    
    print("\nTesting the delivered backdoor model...")
    evaluate_model(backdoor_model, param, clean_testset_loader, backdoor_testset_loader)
    
    # ==================== STAGE 1: LGFD ====================
    print("\n" + "="*80)
    print("STAGE 1: Loss-Guided Feature Decoupling (LGFD)")
    print("="*80)
    
    analysis_results = analyze_and_partition_dataset(backdoor_model, mixed_dataloader, param)
    
    inferred_target = analysis_results['inferred_target']
    refined_suspicion = analysis_results['refined_suspicion_indices']
    refined_trust = analysis_results['refined_trust_indices']
    
    print(f"\nLGFD Results:")
    print(f"  Inferred target label: {inferred_target}")
    print(f"  Refined suspicion set size: {len(refined_suspicion)}")
    print(f"  Refined trust set size: {len(refined_trust)}")
    
    # Create initial partitioned dataloaders for TMFS stage
    all_imgs = analysis_results['all_imgs']
    all_labels = analysis_results['all_labels']
    all_is_bd = analysis_results['all_is_bd']
    
    # Inspection set: suspicious samples (for training auxiliary detector)
    inspection_imgs = all_imgs[refined_suspicion]
    inspection_labels = all_labels[refined_suspicion]
    inspection_dataset = CustomDataset(images=inspection_imgs, labels=inspection_labels)
    inspection_loader = DataLoader(inspection_dataset, batch_size=param.batch_size, shuffle=True)
    
    # Trust set: trusted samples (initial clean set)
    trust_imgs = all_imgs[refined_trust]
    trust_labels = all_labels[refined_trust]
    trust_dataset = CustomDataset(images=trust_imgs, labels=trust_labels)
    trust_loader = DataLoader(trust_dataset, batch_size=param.batch_size, shuffle=True)
    
    print(f"\nInitial partitions created:")
    print(f"  Inspection loader: {len(inspection_dataset)} samples")
    print(f"  Trust loader: {len(trust_dataset)} samples")
    
    # ==================== STAGE 2: TMFS ====================
    print("\n" + "="*80)
    print("STAGE 2: Tri-Model Functional Synergy (TMFS)")
    print("="*80)
    
    # Step 1: Train auxiliary detector (fenh) with uniform label noise
    print("\nStep 1: Training auxiliary detector with uniform label noise...")
    fenh = train_auxiliary_detector_with_uniform_noise(
        backdoor_model, mixed_dataloader, param, num_epochs=5, lr=0.001, noise_rate=0.3
    )
    
    # Save auxiliary detector
    fenh_save_path = os.path.join(save_dir, f"{param.dataset}_{param.attack}_auxiliary_detector.pth")
    torch.save(fenh.state_dict(), fenh_save_path)
    print(f"Auxiliary detector saved: {fenh_save_path}")
    
    # Evaluate auxiliary detector
    print("\nEvaluating auxiliary detector...")
    evaluate_model(fenh, param, clean_testset_loader, backdoor_testset_loader)
    
    # Step 2: Purify the training set using the auxiliary detector
    print("\nStep 2: Purifying training dataset...")
    retained_indices, purified_dataloader = purify_dataset_with_detector(
        fenh, mixed_dataloader, param, loss_threshold=None
    )
    
    print(f"Purified dataset created with {len(purified_dataloader.dataset)} samples")
    
    # Step 3: Train reference model (fref) on purified data
    print("\nStep 3: Training reference model on purified data...")
    fref = train_reference_model(task, purified_dataloader, param, num_epochs=10, lr=0.001)
    
    # Save reference model
    fref_save_path = os.path.join(save_dir, f"{param.dataset}_{param.attack}_reference_model.pth")
    torch.save(fref.state_dict(), fref_save_path)
    print(f"Reference model saved: {fref_save_path}")
    
    # Evaluate reference model
    print("\nEvaluating reference model...")
    evaluate_model(fref, param, clean_testset_loader, backdoor_testset_loader)
    
    # Step 4: Deploy tri-model firewall
    print("\n" + "="*80)
    print("DEPLOYMENT: Tri-Model Firewall")
    print("="*80)
    print("\nFirewall configuration:")
    print("  - fbd: Original delivered model")
    print("  - fenh: Auxiliary detector (uniform noise fine-tuned)")
    print("  - fref: Reference model (trained on purified data)")
    print("\nInference requires at most 3 forward passes per query.")
    
    # Test the tri-model firewall on test data
    print("\nTesting tri-model firewall on test set...")
    test_tri_model_firewall(backdoor_model, fenh, fref, clean_testset_loader, param)
    
    print("\n" + "="*80)
    print("✅ IMMUTRI Defense Pipeline Completed Successfully!")
    print("="*80)


def test_tri_model_firewall(fbd, fenh, fref, test_loader, param):
    """
    Test the tri-model firewall on a test dataset.
    
    Args:
        fbd: Original model
        fenh: Auxiliary detector
        fref: Reference model
        test_loader: Test data loader
        param: Parameter object
    """
    fbd.eval()
    fenh.eval()
    fref.eval()
    
    total_samples = 0
    rejected_count = 0
    correct_predictions = 0
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing firewall"):
            if len(data) == 3:
                imgs, labels, is_bd = data
            else:
                imgs, labels = data
                is_bd = None
            
            imgs = imgs.to(param.device)
            labels = labels.to(param.device)
            batch_size = len(imgs)
            
            # Apply tri-model firewall
            predictions, is_rejected = tri_model_firewall_inference(
                fbd, fenh, fref, imgs, param
            )
            
            # Count rejections
            rejected_count += is_rejected.sum().item()
            
            # Count correct predictions (only for non-rejected samples)
            accepted_mask = ~is_rejected
            if accepted_mask.sum() > 0:
                correct_predictions += (predictions[accepted_mask] == labels[accepted_mask]).sum().item()
            
            total_samples += batch_size
    
    rejection_rate = rejected_count / total_samples
    accuracy_on_accepted = correct_predictions / max(total_samples - rejected_count, 1)
    
    print(f"\nFirewall Performance:")
    print(f"  Total samples: {total_samples}")
    print(f"  Rejected samples: {rejected_count} ({rejection_rate:.2%})")
    print(f"  Accepted samples: {total_samples - rejected_count}")
    print(f"  Accuracy on accepted: {accuracy_on_accepted:.2%}")


# Import CustomDataset and DataLoader
from utils.utils import CustomDataset
from torch.utils.data import DataLoader


if __name__ == '__main__':
    main()


