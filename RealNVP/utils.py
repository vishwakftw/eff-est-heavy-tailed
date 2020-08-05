import numpy as np

from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor

from models import RealNVP
from estimators import (filtering_means,
                        filtering_means_heuristic_1,
                        geometric_median_of_means,
                        norm_removal)


def get_dataset(dataset_name, train=True):
    if dataset_name not in ['cifar10']:
        raise ValueError("Invalid dataset")

    if dataset_name == 'cifar10':
        if train:
            img_transform = Compose([
                RandomHorizontalFlip(),
                ToTensor()
            ])
        else:
            img_transform = Compose([
                ToTensor()
            ])

    if dataset_name == 'cifar10':
        dataset = CIFAR10(root='./data/CIFAR10',
                          download=True,
                          train=train,
                          transform=img_transform)
    return dataset


def get_dataloader(dataset_name, batch_size, train=True):
    dataset = get_dataset(dataset_name, train=train)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True)
    return dataloader


def get_model_constructor(dataset_name):
    if dataset_name == 'cifar10':
        return partial(RealNVP, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
    else:
        raise ValueError(f"Invalid dataset {dataset_name}")


def get_vectorized_gradients(model):
    grad_list = []
    for param in model.parameters():
        grad_list.append(param.grad.detach().clone().flatten())
    return torch.cat(grad_list, dim=0)


def get_mean_estimator(algorithm):
    if algorithm not in ['mean', 'svd_filter', 'one_svd_filter', 'gmom', 'norm']:
        raise ValueError("Invalid algorithm")

    if algorithm == 'mean':
        algo = partial(np.mean, axis=0)
    elif algorithm == 'svd_filter':
        algo = partial(filtering_means, n_discard=5, discard_mode='greedy')
    elif algorithm == 'one_svd_filter':
        algo = partial(filtering_means_heuristic_1, n_discard=5, discard_mode='greedy')
    elif algorithm == 'gmom':
        algo = partial(geometric_median_of_means, num_buckets=5)
    else:
        algo = partial(norm_removal, n_discard=5)
    algo.__name__ = algorithm
    return algo


def gradient_sampler(unaggregated_loss, model):
    gradients = []
    n = unaggregated_loss.size(0)

    for i in range(n):
        model.zero_grad()
        tmp = unaggregated_loss[i]
        if i < n - 1:
            tmp.backward(retain_graph=True)
        else:
            tmp.backward()
        gradients.append(get_vectorized_gradients(model).cpu().numpy())

    gradients = np.vstack(gradients)
    return gradients


def update_grad_attributes(parameters, gradient):
    pointer = 0
    for param in parameters:

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad.data.copy_(gradient[pointer:pointer + num_param].view_as(param))

        # Increment the pointer
        pointer += num_param


def get_optimizer_cons(optim, learning_rate):
    if optim == 'sgd':
        return partial(torch.optim.SGD, lr=learning_rate)
    elif optim == 'momentum':
        return partial(torch.optim.SGD, lr=learning_rate, nesterov=True, momentum=0.9)
    elif optim == 'adam':
        return partial(torch.optim.Adam, lr=learning_rate, betas=(0.5, 0.999))
    else:
        raise NotImplementedError
