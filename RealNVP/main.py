import utils
import train

import os
import torch
import numpy as np
from functools import partial
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument('--dataset', type=str, required=True, help='Dataset name')
p.add_argument('--num_iters', type=int, default=5000,
               help='Number of iterations to run for')
p.add_argument('--save_dir', type=str, default='models',
               help='Directory to save the models')
p.add_argument('--algos', nargs='+', type=str,
               help='List of algorithms you want to run')
p.add_argument('--global_seed', type=int, default=1729,
               help='Global seed for NumPy (to determine seeds)')
p.add_argument('--optim', type=str, choices=['sgd', 'momentum', 'adam'],
               help='Optimizer to use')
p.add_argument('--cuda', action='store_true', help='Toggle to use CUDA')
p.add_argument('--n_discard', type=int, default=5,
               help='Number of points to discard')
p.add_argument('--num_save', type=int, default=20,
               help='Number of models to save evenly spaced during training')
p.add_argument('--max_grad_norm', type=float, default=0.,
               help='Maximum gradient norm for clipping')
args = p.parse_args()

BATCH_SIZE = 64  # Batch size

real_nvp_cons = utils.get_model_constructor(args.dataset)  # Get the constructor
train_dataloader = utils.get_dataloader(args.dataset, BATCH_SIZE, train=True)  # Get the dataloader

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)  # Create save directory for saving the models

np.random.seed(args.global_seed)
seed = np.random.randint(0, 5000, args.num_seeds)  # Get seed

sample_mean = partial(np.mean, axis=0)
sample_mean.__name__ = 'mean'

CHECKPOINTS = set(np.linspace(0, args.num_iters, args.num_save + 1, dtype=int))  # 20 models saved
LR = 1e-03  # Learning rate
ALPHA = 0.75  # Streaming weight (specific to streaming approximation)
DEVICE = 'cuda' if args.cuda else 'cpu'
WEIGHT_DECAY = 5e-05

torch.manual_seed(seed)
# Initialize and save
real_nvp = real_nvp_cons()

torch.save(real_nvp.state_dict(),
           f"{args.save_dir}/real_nvp_init_{args.dataset}_{seed}.pt")
del real_nvp

for algo in args.algos:
    print(f"Running seed {seed}, algo {algo}, optim {args.optim}")
    SUFFIX = f"{args.dataset}_{seed}_1_{args.optim}"
    real_nvp = real_nvp_cons()
    real_nvp.load_state_dict(torch.load(f"{args.save_dir}/real_nvp_init_{args.dataset}_{seed}.pt"))

    if args.cuda:
        real_nvp = real_nvp.cuda()

    # Train with each algorithm
    if algo in ['mean']:
        mean_est = utils.get_mean_estimator(algo)
        train.plugin_estimator_training_loop(real_nvp, train_dataloader, LR,
                                             args.optim, DEVICE, args.num_iters,
                                             CHECKPOINTS, BATCH_SIZE, mean_est,
                                             WEIGHT_DECAY, args.max_grad_norm,
                                             args.save_dir, SUFFIX)
    elif algo == 'streaming_approx':
        train.streaming_approx_training_loop(real_nvp, train_dataloader, LR,
                                             args.optim, DEVICE, args.num_iters,
                                             CHECKPOINTS, ALPHA, BATCH_SIZE,
                                             args.n_discard, WEIGHT_DECAY,
                                             args.max_grad_norm, args.save_dir, SUFFIX)
