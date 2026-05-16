from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict
import logging
import torch
logger = logging.getLogger('logger')

ALL_TASKS =  ['backdoor', 'normal', 'sentinet_evasion', #'spectral_evasion',
                           'neural_cleanse', 'mask_norm', 'sums', 'neural_cleanse_part1']

@dataclass
class Params:

    # Corresponds to the class module: tasks.mnist_task.MNISTTask
    # See other tasks in the task folder.
    task: str = 'MNIST'

    current_time: str = None
    name: str = None
    commit: float = None
    random_seed: int = None
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # training params
    start_epoch: int = 1
    epochs: int = None
    log_interval: int = 1000

    # model arch is usually defined by the task
    pretrained: bool = False
    resume_model: str = None
    lr: float = None
    decay: float = None
    momentum: float = None
    optimizer: str = None
    scheduler: bool = False
    scheduler_milestones: List[int] = None
    # data
    data_path: str = '.data/'
    batch_size: int = 64
    test_batch_size: int = 100
    transform_train: bool = True
    "Do not apply transformations to the training images."
    max_batch_id: int = None
    "For large datasets stop training earlier."
    input_shape = None
    "No need to set, updated by the Task class."

    # gradient shaping/DP params
    dp: bool = None
    dp_clip: float = None
    dp_sigma: float = None

    # attack params
    backdoor: bool = False
    backdoor_label: int = 8
    poisoning_proportion: float = 1.0  # backdoors proportion in backdoor loss
    synthesizer: str = 'pattern'
    backdoor_dynamic_position: bool = False

    # losses to balance: `normal`, `backdoor`, `neural_cleanse`, `sentinet`,
    # `backdoor_multi`.
    loss_tasks: List[str] = None

    loss_balance: str = 'MGDA'
    "loss_balancing: `fixed` or `MGDA`"

    loss_threshold: float = None

    # approaches to balance losses with MGDA: `none`, `loss`,
    # `loss+`, `l2`
    mgda_normalize: str = None
    fixed_scales: Dict[str, float] = None

    # relabel images with poison_number
    poison_images: List[int] = None
    poison_images_test: List[int] = None
    # optimizations:
    alternating_attack: float = None
    clip_batch: float = None
    # Disable BatchNorm and Dropout
    switch_to_eval: float = None

    # nc evasion
    nc_p_norm: int = 1
    # spectral evasion
    spectral_similarity: 'str' = 'norm'

    # logging
    report_train_loss: bool = True
    log: bool = True
    tb: bool = True
    save_model: bool = True
    save_on_epochs: List[int] = False
    save_scale_values: bool = True
    print_memory_consumption: bool = True
    save_timing: bool = False
    timing_data = None

    # Temporary storage for running values
    running_losses = None
    running_scales = None

    # FL params
    fl: bool = False
    fl_no_models: int = 100
    fl_local_epochs: int = 2
    fl_total_participants: int = 80000
    fl_eta: int = 1
    fl_sample_dirichlet: bool = False
    fl_dirichlet_alpha: float = None
    fl_diff_privacy: bool = False
    fl_dp_clip: float = None
    fl_dp_noise: float = None
    # FL attack details. Set no adversaries to perform the attack:
    fl_number_of_adversaries: int = 0
    fl_single_epoch_attack: int = None
    fl_weight_scale: int = 1

    def __post_init__(self):
        # enable logging anyways when saving statistics
        if self.save_model or self.tb or self.save_timing or \
                self.print_memory_consumption:
            self.log = True

        if self.log:
            self.folder_path = f'saved_models/model_' \
                               f'{self.task}_{self.current_time}_{self.name}'

        self.running_losses = defaultdict(list)
        self.running_scales = defaultdict(list)
        self.timing_data = defaultdict(list)

        for t in self.loss_tasks:
            if t not in ALL_TASKS:
                raise ValueError(f'Task {t} is not part of the supported '
                                 f'tasks: {ALL_TASKS}.')

    def to_dict(self):
        return asdict(self)
    
class My_Params():
    def __init__(self):
        self.optimizer = 'SGD'
        self.scheduler = False
        self.batch_size = 128
        self.test_batch_size = 128
        self.resume_model = False
        self.transform_train = True
        self.data_path = '/root/autodl-fs/datasets/'
        self.poison_images: List[int] = None
        self.poison_images_test: List[int] = None
        self.pretrained = False
        self.lr = 0.001
        self.decay = 0.99
        self.momentum = 0.9
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.backdoor_dynamic_position = False
        self.input_shape = None
        
        
def set_param(param, args):
    param.data_path = args.data_path
    param.dataset = args.dataset
    param.attack = args.attack
    param.batch_size = args.batch_size
    param.target = args.target
    param.num_classes = 10
    if param.dataset == 'gtsrb':
        param.num_classes = 43