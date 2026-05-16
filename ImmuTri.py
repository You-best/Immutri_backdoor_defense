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
from utils.utils import evaluate, evaluate_model, confusion_mislabel
from core.feature_analyzer import analyze_and_partition_dataset
from core.data_purification import create_purified_partitions, get_filtered_data


def main():
    task = get_task(param)

    # 获取对应的综合器
    synthesizer = applied_attack(param, task)
    # 获取对应的数据集
    clean_testset_loader, backdoor_testset_loader, \
                mixed_dataloader = get_dataloader(param, task, synthesizer)

    backdoor_model = task.build_model().to(param.device)

    save_dir = './save'
    os.makedirs(save_dir, exist_ok=True)
    backdoor_model_name = f"{param.dataset}_{param.attack}_backdoor_model.pth"
    backdoor_model_save_path = os.path.join(save_dir, backdoor_model_name)
    backdoor_model_save_path = '/home/star/sda1/backdoors101-master/ImmuTri/save/cifar10_badnets_backdoor_model.pth'
    backdoor_model.load_state_dict(torch.load(backdoor_model_save_path))
    
    print("begin to test the backdoor model...")
    evaluate_model(backdoor_model, param, clean_testset_loader, backdoor_testset_loader)
    analysis_results = analyze_and_partition_dataset(backdoor_model, mixed_dataloader, param)

    pipeline_config = {
        'similarity_threshold': 0.56,
        'candidate_index_threshold': 300,
        'inspection_set_ratio': 0.1,
        'clean_set_ratio': 0.5,
        'batch_size': param.batch_size,
    }

    # 4. Call the main orchestrator function
    inspection_loader, clean_loader = create_purified_partitions(
        analysis_results, 
        param, 
        pipeline_config
    )
    print(f"clean_set_dataloader的length：{len(clean_loader)}")
    print(f"inspection_set_dataloader的length：{len(inspection_loader)}")

    sensitive_model = task.build_model().to(param.device)
    # train_with_confusion_regularization(sensitive_model, inspection_loader, clean_loader, param, num_epochs=6)
    train_with_confusion_regularization(
        sensitive_model, 
        inspection_loader, 
        clean_loader, 
        param,
        clean_testset_loader=clean_testset_loader,     
        backdoor_testset_loader=backdoor_testset_loader,     
        num_epochs=6
    )
    sensitive_model_save_dir = './save'
    sensitive_model_name = f"{param.dataset}_{param.attack}_sensitive_model.pth"
    sensitive_model_save_path = os.path.join(sensitive_model_save_dir, sensitive_model_name)
    torch.save(sensitive_model.state_dict(), sensitive_model_save_path)
    print(f"sensitive model saved: {sensitive_model_save_path}")
    
    loss_threshold = 0.2
    filtered_idxs = get_filtered_data(sensitive_model, mixed_dataloader, loss_threshold, param)
    

def train_with_confusion_regularization(
    model, 
    trusted_dataloader, 
    confusion_dataloader, 
    param, 
    num_epochs=10, 
    learning_rate=0.001, 
    confusion_weight=0.8,
    clean_testset_loader=None,
    backdoor_testset_loader=None
):
    """
    Trains a model using a trusted dataset and applies confusion-based regularization.
    """
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create a persistent iterator for the confusion dataloader
    confusion_iter = iter(confusion_dataloader)

    # Start the training loop
    for epoch in range(num_epochs):
        model.train()
        
        progress_bar = tqdm(
            enumerate(trusted_dataloader), 
            desc=f'Epoch {epoch+1}/{num_epochs}', 
            total=len(trusted_dataloader)
        )
        
        for batch_idx, (trusted_images, trusted_labels) in progress_bar:
            trusted_images = trusted_images.to(param.device)
            trusted_labels = trusted_labels.to(param.device)

            try:
                confusion_images, original_confusion_labels = next(confusion_iter)
            except StopIteration:
                confusion_iter = iter(confusion_dataloader)
                confusion_images, original_confusion_labels = next(confusion_iter)

            confusion_images = confusion_images.to(param.device)
            original_confusion_labels = original_confusion_labels.to(param.device)
            
            # --- Generate Mislabeled Targets ---
            # Generate confusing labels for the reserved batch
            mislabeled_targets = confusion_mislabel(
                original_confusion_labels, 
                num_classes=param.num_classes, 
                confusion_strength=2, 
                epoch=epoch, 
                batch_idx=batch_idx
            )
            
            # --- Calculate Losses ---
            trusted_set_loss = criterion(model(trusted_images), trusted_labels)
            
            # Confusion loss on the reserved data with mislabeled targets
            confusion_set_loss = criterion(model(confusion_images), mislabeled_targets)

            # --- Combine Losses ---
            # The total loss is a weighted sum of the trusted loss and the confusion loss
            combined_loss = trusted_set_loss + confusion_weight * confusion_set_loss

            # --- Backpropagation ---
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

        # --- Post-Epoch Evaluation ---
        print(f"\nEpoch [{epoch+1}/{num_epochs}] finished.")
        evaluate_model(model, param, clean_testset_loader, backdoor_testset_loader)

if __name__ == '__main__':
    main()


