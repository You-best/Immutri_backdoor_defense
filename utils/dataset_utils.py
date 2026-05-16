import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.notebook import tqdm
class SubsetDataset(Dataset):
    def __init__(self, original_dataset, r, device, synthesizer=None, target=None, only_posion_target=False, Save_sample=False, clean_label=True, attack_params=None,
                rm_tar=False):
        """
        Args:
            original_dataset (Dataset): The original dataset instance.
            r (float): The proportion of data to be extracted (0 < r <= 1).
        """
        self.original_dataset = original_dataset
        self.r = r
        self.target = target
        self.params = attack_params
        self.device = device
        self.Save_sample = Save_sample
        self.clean_label = clean_label
        self.only_posion_target = only_posion_target
        self.rm_tar = rm_tar
        if only_posion_target:
            self.indices = self.select_by_target(target, r)
        else:
            self.indices = self._get_subset_indices()
        self.synthesizer = synthesizer
        if self.Save_sample:
            self.original_dataset = self.save_sample()

    def _get_subset_indices(self):
        """根据比例 `r` 随机选择子集索引。"""
        total_size = len(self.original_dataset)

        # 如果 rm_tar 为 True，提取出标签不等于 self.target 的样本
        if self.rm_tar:
            # 获取所有标签
            all_labels = [self.original_dataset[i][1] for i in range(total_size)]
            # 提取标签不等于 self.target 的样本索引
            remaining_indices = [i for i, label in enumerate(all_labels) if label != self.target]
            remaining_size = len(remaining_indices)
            # 根据比例 r 选择子集
            subset_size = int(remaining_size * self.r)
            subset_indices = torch.randperm(remaining_size)[:subset_size].tolist()
            return [remaining_indices[i] for i in subset_indices]
        else:
            # 正常情况，随机选择子集
            subset_size = int(total_size * self.r)
            return torch.randperm(total_size)[:subset_size].tolist()

    
    def select_by_target(self, target, r):
        # 先选择出所有标签为 target 的样本下标
        target_indices = [i for i, (_, label) in enumerate(self.original_dataset) if label == target]
        # 计算子集大小
        subset_size = int(len(target_indices) * r)
        # 随机打乱并选择前 subset_size 个下标
        selected_indices = torch.randperm(len(target_indices))[:subset_size].tolist()
        # 返回挑选的下标
        return [target_indices[i] for i in selected_indices]
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get item from the original dataset using the saved indices."""
        original_idx = self.indices[idx]
        input_, label = self.original_dataset[original_idx]
        input_, label = input_.to(self.device), label
        if self.synthesizer and not self.Save_sample:
            input_ = self.synthesizer.apply_backdoor_to_a_sample(data=input_, label=label, params=self.params)
            if not self.clean_label:
                label = self.target
        return (input_, label)
    
    def set_attack_params(self, params=None):
        self.params = params
    
    def save_sample(self, batch_size=32):
        """
        按批次生成并保存样本，batch_size 默认为 64。
        """
        # 预先创建保存样本的列表
        print("开始生成并保存样本....")
        saved_sample = [(None, None) for i in range(len(self.original_dataset))]

        # 按批次处理 self.indices
        num_batches = len(self.indices) // batch_size + (1 if len(self.indices) % batch_size > 0 else 0)

        for batch_idx in tqdm(range(num_batches)):
            # 计算当前批次的开始和结束索引
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.indices))

            # 获取该批次的索引
            batch_indices = self.indices[start_idx:end_idx]

            # 获取批次数据
            inputs, labels = zip(*[self.original_dataset[idx] for idx in batch_indices])
            with torch.no_grad():
                inputs_list = []
                for input_ in inputs:
                    inputs_list.append(input_.to(self.device))
                inputs = torch.stack(inputs_list)
                
            # inputs = torch.stack([input_.to(self.device) for input_ in inputs])  # 转换为张量并移动到设备
            labels = torch.tensor(labels).to(self.device)

            # 应用背门攻击（如果有 synthesizer）
            if self.synthesizer:
                inputs = self.synthesizer.apply_backdoor_to_a_sample(data=inputs, label=labels, params=self.params)
            if not self.clean_label:
                labels = torch.full((inputs.size(0),), self.target, dtype=torch.long).to(self.device)

            # 将批次数据保存到 saved_sample 中
            for i, idx in enumerate(batch_indices):
                saved_sample[idx] = (inputs[i].cpu(), labels[i].cpu().item())  # 将数据移动回 CPU 以节省 GPU 内存

        return saved_sample



class CombinedDataset(Dataset):
    def __init__(self, clean_set, backdoor_set, device, replace=False):
        """
        Args:
            clean_set (Dataset): The first dataset instance (e.g., clean dataset).
            backdoor_set (Dataset): The second dataset instance (e.g., backdoor dataset).
        """
        self.clean_set = clean_set
        self.backdoor_set = backdoor_set
        self.clean_set_len = len(clean_set)
        self.backdoor_set_len = len(backdoor_set)
        self.backdoor_indies = backdoor_set.indices
        self.total_len = self.clean_set_len + self.backdoor_set_len
        self.device = device
        self.replace = replace  # Whether to replace samples based on backdoor indices

    def __len__(self):
        """Return the total number of samples in the combined dataset."""
        if not self.replace:
            return self.total_len
        else:
            return self.clean_set_len

    def __getitem__(self, idx):
        """Retrieve the item from the corresponding dataset based on the index."""
        if not self.replace:
            if idx < self.clean_set_len:
                return (self.clean_set[idx][0].to(self.device), self.clean_set[idx][1], 0)
            else:
                # Adjust index for the second dataset
                adjusted_idx = idx - self.clean_set_len
                return (self.backdoor_set[adjusted_idx][0].to(self.device), self.backdoor_set[adjusted_idx][1], 1)
        else:
            # If replace is True, check if idx corresponds to a backdoor sample index
            if idx in self.backdoor_indies:
                adjusted_idx = self.backdoor_indies.index(idx)  # Get the index in backdoor_set
                return (self.backdoor_set[adjusted_idx][0].to(self.device), self.backdoor_set[adjusted_idx][1], 1)
            else:
                # If idx is not in backdoor_indies, return from clean_set
                return (self.clean_set[idx][0].to(self.device), self.clean_set[idx][1], 0)