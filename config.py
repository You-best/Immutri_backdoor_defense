import argparse
import torch
from typing import List

def parser_handle():
    '''
    Input hyperparameters
    '''
    parser = argparse.ArgumentParser(description='Parameters for Model Training and Attacks')

    parser.add_argument('--data_path', type=str, default='/home/star/sda1/data/1744ecab-59b5-4839-9124-a5d97b52e660/datasets/cifar10',
                        help='Path to the dataset directory.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Input batch size for training.')
    parser.add_argument('--target', type=int, default=8,
                        help='poisonous target label.')
    parser.add_argument('--attack', type=str, default='badnets',
                        help='The type of poisoning attack (e.g., "BadNets").')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id.')

    return parser

