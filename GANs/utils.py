import numpy as np

from functools import partial

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose

import archs


def get_dataset(dataset_name, train=True):
    if dataset_name not in ['mnist', 'cifar10']:
        raise ValueError("Invalid dataset")

    if dataset_name == 'mnist':
        img_transform = Compose([
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
    else:
        img_transform = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if dataset_name == 'mnist':
        dataset = MNIST(root='./data/MNIST',
                        download=True,
                        train=train,
                        transform=img_transform)
    else:
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


def get_model_constructors(dataset_name):
    if dataset_name == 'mnist':
        return archs.GeneratorDCGANMNIST, archs.DiscriminatorDCGANMNIST
    elif dataset_name == 'cifar10':
        return archs.GeneratorDCGANCIFAR10, archs.DiscriminatorDCGANCIFAR10
    else:
        raise ValueError(f"Invalid dataset {dataset_name}")


def get_vectorized_gradients(generator_obj):
    grad_list = []
    for param in generator_obj.parameters():
        grad_list.append(param.grad.detach().clone().flatten())
    return torch.cat(grad_list, dim=0)


def gradient_sampler(discriminator_obj, generator_obj,
                     loss_fn, noise_batch=None,
                     noise_batchsize=None, latent_dim=None,
                     device=None, compute_norms=False):
    if noise_batch is None:
        noise_batch = torch.randn(noise_batchsize, latent_dim, device=device)

    discriminator_obj.zero_grad()
    generator_obj.zero_grad()
    fake_images = generator_obj(noise_batch)
    fake_preds = discriminator_obj(fake_images).squeeze()
    gen_loss = loss_fn(fake_preds)

    gradients = []
    n = noise_batch.size(0)
    for i in range(0, n):
        discriminator_obj.zero_grad()
        generator_obj.zero_grad()
        tmp = gen_loss[i]
        if i < n - 1:
            tmp.backward(retain_graph=True)
        else:
            tmp.backward()
        if not compute_norms:
            gradients.append(get_vectorized_gradients(generator_obj).cpu().numpy())
        else:
            with torch.no_grad():
                total_norm = 0.0
                for param in generator_obj.parameters():
                    total_norm += param.grad.norm().item() ** 2
            gradients.append(np.sqrt(total_norm))

    if not compute_norms:
        gradients = np.vstack(gradients)
    return gradients


def get_mean_estimator(algorithm):
    if algorithm not in ['mean']:
        raise ValueError("Invalid algorithm")

    if algorithm == 'mean':
        algo = partial(np.mean, axis=0)
    algo.__name__ = algorithm
    return algo


def get_optimizer_cons(optim, learning_rate):
    if optim == 'sgd':
        return partial(torch.optim.SGD, lr=learning_rate)
    elif optim == 'momentum':
        return partial(torch.optim.SGD, lr=learning_rate, nesterov=True, momentum=0.9)
    elif optim == 'adam':
        return partial(torch.optim.Adam, lr=learning_rate, betas=(0.5, 0.999))
    else:
        raise NotImplementedError


def update_grad_attributes(parameters, gradient):
    pointer = 0
    for param in parameters:

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad.data.copy_(gradient[pointer:pointer + num_param].view_as(param))

        # Increment the pointer
        pointer += num_param
