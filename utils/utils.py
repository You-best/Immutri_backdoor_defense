import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None): #, is_bd
        """
        初始化数据集。
        
        参数:
        - images: 图像数据，可以是路径列表或直接是图像数据数组
        - labels: 图像对应的标签
        - is_bd: 是否为特定样本的标志
        - transform: 图像预处理操作（数据增强），可选
        """
        self.images = images
        self.labels = labels
        # self.is_bd = is_bd
        self.transform = transform
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。
        
        参数:
        - idx: 数据索引
        
        返回:
        - 样本数据，包括图像、标签和is_bd
        """
        # 如果是图像路径，加载图像
        image = self.images[idx]
        if isinstance(image, str):  # 如果传入的是文件路径
            image = Image.open(image).convert('RGB')
        
        # 应用数据增强（如果有的话）
        if self.transform:
            image = self.transform(image)
        
        # 获取标签和is_bd标志
        label = self.labels[idx]
        # is_bd = self.is_bd[idx]
        
        return image, label #, is_bd

def evaluate(model, dataloader, criterion, device):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():  # 关闭梯度计算
        for data in tqdm(dataloader, desc="Evaluating Model"):
            # Handle the case where data has 2 or 3 elements
            if len(data) == 2:
                imgs, labels = data
            else:
                imgs, labels, _ = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    model.train()  # 重新设置模型为训练模式
    return avg_loss, accuracy

def evaluate_model(model, param, clean_testset_loader, backdoor_testset_loader):
    criterion = nn.CrossEntropyLoss() # SCELoss(alpha=1.0, beta=0.5, num_classes=len(task.classes))# nn.CrossEntropyLoss()
    model = model.to(param.device)
    avg_clean_loss, clean_acc = evaluate(model, clean_testset_loader, criterion,  param.device)
    avg_backdoor_loss, backdoor_acc = evaluate(model, backdoor_testset_loader, criterion,  param.device)
    print("Deep:\nclean_acc:{} , backdoor_acc:{}".format(clean_acc,backdoor_acc))
    print("avg_clean_loss:{} , avg_backdoor_loss:{}\n".format(avg_clean_loss,avg_backdoor_loss))
    
def detect_malicious_threshold(individual_losses_np, bandwidth=0.05, plot=False):
    """
    Detect the threshold for malicious samples with an adaptive mechanism to filter out outliers.
    
    Parameters:
    individual_losses_np (numpy array): The loss values for each sample.
    
    Returns:
    float: The threshold value for detecting malicious samples.
    numpy array: Indices of samples considered malicious after adaptive filtering.
    """
    # 将损失值转换为二维数组以便 KDE 处理
    individual_losses_reshaped = individual_losses_np.reshape(-1, 1)

    # 进行核密度估计
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(individual_losses_reshaped)
    log_density = kde.score_samples(individual_losses_reshaped)

    # 对损失值进行排序
    sorted_losses = np.sort(individual_losses_np)
    log_density_sorted = kde.score_samples(sorted_losses.reshape(-1, 1))
    
    # 可视化排序后的概率密度分布函数
    if plot:
        plt.figure(figsize=(5, 3))
        plt.plot(sorted_losses, np.exp(log_density_sorted), 'b-', label='Estimated Density')
        plt.fill_between(sorted_losses, np.exp(log_density_sorted), alpha=0.5)
        plt.xlabel('Loss Value')
        plt.ylabel('Density')
        plt.title('Kernel Density Estimation of Sorted Loss Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 获取密度估计的实际值
    density_values = np.exp(log_density_sorted)

    # 找到所有的局部峰值和谷值（最小值）
    peaks, _ = find_peaks(density_values)
    valleys, _ = find_peaks(-density_values)

    # 找到第一个峰值（波峰）
    first_peak_index = peaks[0]

    # 在第一个波峰之后，找到第一个谷值（波谷）
    threshold_index = None
    for valley in valleys:
        if valley > first_peak_index:
            threshold_index = valley
            break

    # 如果没有找到有效的谷值，将阈值设置为0.1
    if threshold_index is None:
        final_threshold = 0.1
    else:
        # 根据索引获取初步的阈值
        initial_threshold = sorted_losses[threshold_index]

        # 初步筛选出低于该阈值的样本
        preliminary_malicious_indices = np.where(individual_losses_np <= initial_threshold)[0]
        preliminary_malicious_losses = individual_losses_np[preliminary_malicious_indices]
        preliminary_malicious_losses.sort()

        # 计算相邻损失值之间的差异，并排除最大和最小值
        differences = np.diff(preliminary_malicious_losses)
        if len(differences) > 2:
            filtered_differences = np.sort(differences)[1:-1]  # 排除最大和最小差异
        else:
            filtered_differences = differences

        # 设定自适应阈值，排除那些与前一个样本差异超过平均差异10倍的值
        mean_diff = np.mean(filtered_differences)
        adaptive_threshold_index = np.where(differences > 100 * mean_diff)[0]

        if len(adaptive_threshold_index) > 0:
            adaptive_threshold_index = adaptive_threshold_index[0] + 1  # 加1是因为差分数组比原数组少一个元素
            filtered_losses = preliminary_malicious_losses[:adaptive_threshold_index]
            final_threshold = filtered_losses[-1]
        else:
            filtered_losses = preliminary_malicious_losses
            final_threshold = initial_threshold

        # if final_threshold > 1.0:    
        #     final_threshold = 0

    # 根据最终的自适应阈值筛选恶意样本
    malicious_indices = None
    if final_threshold != 0:
        malicious_indices = np.where(individual_losses_np <= final_threshold)[0]
        
    return final_threshold, malicious_indices    
    
    
def get_cosine_similarity_matrix(featuredata, b_new_list, param):
    feature_tensor = torch.stack([featuredata[i][0] for i in range(len(featuredata))])
    b_new_tensor = torch.stack(b_new_list)
    # cosine_similarity = F.cosine_similarity(feature_tensor, b_new_tensor.T)
    if not isinstance(feature_tensor, torch.Tensor):
        feature_tensor = torch.tensor(feature_tensor).to(param.device)
    else:
        feature_tensor = feature_tensor.clone().detach().to(param.device)

    if not isinstance(b_new_tensor, torch.Tensor):
        b_new_tensor = torch.tensor(b_new_tensor).to(param.device)
    else:
        b_new_tensor = b_new_tensor.clone().detach().to(param.device)
    # 步骤 1: 规范化 A 和 B
    feature_tensor_normalized = feature_tensor / feature_tensor.norm(dim=1, keepdim=True)  # [65000, 2]
    b_new_tensor_normalized = b_new_tensor / b_new_tensor.norm(dim=1, keepdim=True)  # [1000, 2]
    
    # 步骤 2: 计算余弦相似度矩阵
    # 结果的形状是 [65000, 1000]
    cosine_similarity_matrix = torch.matmul(feature_tensor_normalized, b_new_tensor_normalized.T)
    return cosine_similarity_matrix
    
def confusion_mislabel(
    original_labels, 
    num_classes=10, 
    confusion_strength=1, 
    epoch=0, 
    batch_idx=0
):
    """
    Generates mislabeled targets based on the original labels.
    Returns:
        torch.Tensor: A tensor of newly generated mislabeled targets.
    """
    # Clone the original labels to avoid modifying them in-place
    mislabeled_targets = original_labels.clone()
    
    for i in range(original_labels.size(0)):
        original_label = original_labels[i].item()

        # --- Generate a candidate for the new, incorrect label ---

        # 1. Create a base random offset that is never zero.
        # This ensures the new label is at least different from the original in a simple case.
        base_random_offset = random.randint(1, num_classes - 1)
        
        # 2. Add a dynamic component based on training progress.
        # This makes the mislabeling pattern change over time.
        dynamic_shift = epoch * len(original_labels) + batch_idx # A simple way to make it dynamic
        
        # 3. Combine and wrap around the number of classes to get a candidate label.
        candidate_label = (original_label + base_random_offset + dynamic_shift) % num_classes
        
        # --- Final check to guarantee the new label is different ---
        # In the rare case that the candidate label matches the original, simply increment it.
        if candidate_label == original_label:
            candidate_label = (candidate_label + 1) % num_classes

        mislabeled_targets[i] = candidate_label

    return mislabeled_targets


