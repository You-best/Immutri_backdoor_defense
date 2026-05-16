from tasks.imagenet10_task import Imagenet10Task
from tasks.cifar10_task import Cifar10Task
from tasks.gtsrb_task import GtsrbTask

from attacks.badnets_synthesizer import BadnetsSynthesizer
from attacks.blend_synthesizer import BlendSynthesizer
from attacks.CL_synthesizer import CLSynthesizer
from attacks.inputaware_synthesizer import InputAwareSynthesizer
from attacks.wanet_synthesizer import WaNetSynthesizer
import logging

from .dataset_utils import CombinedDataset, SubsetDataset
from torch.utils.data import Dataset, DataLoader, Subset


def get_task(param):
    task = None
    if param.dataset == 'cifar10':
        task = Cifar10Task(param)
    elif param.dataset == 'gtsrb':
        task = GtsrbTask(param)
    elif param.dataset == 'imagenet10':
        task = Imagenet10Task(param)
    else:
        raise ValueError(
            f"Unsupported dataset '{param.dataset}'. "
        ) from None
    return task

def applied_attack(param, task):

    attack_name = param.attack.lower()
    
    # logging.info(f"Initializing attack synthesizer for: '{attack_name}'")
    print(f"Initializing attack synthesizer for: '{attack_name}'")

    if attack_name == 'badnets':
        synthesizer = BadnetsSynthesizer(task)
    elif attack_name == 'blend':
        synthesizer = BlendSynthesizer(task)
    elif attack_name == 'cl':
        synthesizer = CLSynthesizer(task)
    elif attack_name == 'inputaware':
        synthesizer = InputAwareSynthesizer(task, dataset=param.dataset)
    elif attack_name == 'wanet':
        synthesizer = WaNetSynthesizer(task, dataset=param.dataset)
    else:
        supported_attacks = ['badnets', 'blend', 'bpp', 'cl', 'inputaware', 'wanet']
        raise ValueError(f"Unknown or unsupported attack: '{param.attack}'. "
                         f"Supported attacks are: {supported_attacks}")
    
    return synthesizer
    
def get_dataloader(param, task, synthesizer):
    attack_name = param.attack.lower()
    prate = 0.1
    
    target = param.target
    if attack_name in ['cl', 'wanet']:
        prate = 0.3
    
    if attack_name == 'cl':
        attack_params = {'adv':True}
        backdoor_set = SubsetDataset(original_dataset=task.train_dataset, r=prate, 
                                     device=param.device, 
                                     synthesizer=synthesizer, target=target, 
                                     only_posion_target=True, Save_sample=True, 
                                     clean_label=True, attack_params=attack_params)
    else:
        backdoor_set = SubsetDataset(original_dataset=task.train_dataset, r=prate, 
                                     device=param.device, 
                                     synthesizer=synthesizer, target=target,
                                     only_posion_target=False, Save_sample=False, 
                                     clean_label=False, attack_params=None, rm_tar=False)
        
    bd_trainingset = SubsetDataset(original_dataset=task.train_dataset, r=1.0, 
                                   device=param.device, synthesizer=synthesizer, target=target,
                                only_posion_target=False, Save_sample=False, 
                                clean_label=False, attack_params=None, rm_tar=False)
    bd_testset = SubsetDataset(original_dataset=task.test_dataset, r=1.0, 
                               device=param.device, synthesizer=synthesizer, target=target,
                                only_posion_target=False, Save_sample=False, 
                                clean_label=False, attack_params=None, rm_tar=False)

    backdoor_dataset = CombinedDataset(clean_set=task.train_dataset, backdoor_set=backdoor_set, device=param.device)

    batch_size = param.batch_size
    clean_testset_loader = DataLoader(dataset=task.test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    backdoor_testset_loader = DataLoader(dataset=bd_testset, batch_size=batch_size, shuffle=True, num_workers=0)

    mixed_dataloader = DataLoader(dataset=backdoor_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return clean_testset_loader, backdoor_testset_loader, mixed_dataloader
