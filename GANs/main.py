import utils
import train

import os
import torch
import numpy as np
from functools import partial
from copy import deepcopy
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument('--dataset', type=str, required=True, help='Dataset name')
p.add_argument('--num_iters', type=int, default=50000,
               help='Number of iterations to run for')
p.add_argument('--save_dir', type=str, default='models',
               help='Directory to save the models')
p.add_argument('--algos', nargs='+', type=str,
               help='List of algorithms you want to run')
p.add_argument('--global_seed', type=int, default=1729,
               help='Global seed for NumPy (to determine seeds)')
p.add_argument('--loss_fn', type=str, choices=['standard', 'logD'],
               help='Loss function to use')
p.add_argument('--optim', type=str, choices=['sgd', 'momentum', 'adam'],
               help='Optimizer to use')
p.add_argument('--cuda', action='store_true', help='Toggle to use CUDA')
p.add_argument('--n_discard', type=str, default='5',
               help='Number of points to discard')
args = p.parse_args()

BATCH_SIZE = 64  # Batch size

gen_cons, disc_cons = utils.get_model_constructors(args.dataset)  # Get the constructors
train_dataloader = utils.get_dataloader(args.dataset, BATCH_SIZE, train=True)  # Get the dataloader

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)  # Create save directory for saving the models

np.random.seed(args.global_seed)
seed = np.random.randint(0, 5000)

sample_mean = partial(np.mean, axis=0)
sample_mean.__name__ = 'mean'


# Parse n_discard - create a callable that return n_discard based on iterations
# Format: comma separated intervals - 2000:1,4000:2,6000:3
# which indicates for iteration number <= 2000 use val 1, <= 4000 use val 2, <= 6000 use val 3
# if no commas, then it's a constant value
class NDiscardSchedule():
    def __init__(self, raw_string, max_iter):
        if raw_string.find(',') == -1:
            self.map_vals = {max_iter: int(raw_string)}
        else:
            self.map_vals = {}
            for interval in raw_string.split(','):
                max_iter, val = interval.split(':')
                self.map_vals[int(max_iter)] = int(val)

    def __call__(self, itr):
        for interval, val in self.map_vals.items():
            if itr <= interval:
                return val
        return val  # default to the last


N_DISCARD = NDiscardSchedule(args.n_discard, args.num_iters)

CHECKPOINTS = set(np.linspace(0, args.num_iters, 21, dtype=int))  # 20 models saved
LATENT_DIM = 128  # Latent dimension
LR = 2e-04  # Learning rate
ALPHA = 0.75  # Streaming weight (specific to streaming approximation)
DEVICE = 'cuda' if args.cuda else 'cpu'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


torch.manual_seed(seed)
# Initialize and save
gen_init, disc_init = gen_cons(LATENT_DIM), disc_cons()
gen_init.apply(weights_init)  # Apply initialization recommended in DCGAN
disc_init.apply(weights_init)  # Apply initialization recommended in DCGAN

torch.save(gen_init.state_dict(),
           f"{args.save_dir}/generator_init_{args.dataset}_{seed}.pt")
torch.save(disc_init.state_dict(),
           f"{args.save_dir}/discriminator_init_{args.dataset}_{seed}.pt")

for algo in args.algos:
    print(f"Running seed {seed}, algo {algo}, loss {args.loss_fn}, optim {args.optim}")
    SUFFIX = f"{args.dataset}_{seed}_1_{args.loss_fn}_{args.optim}_{args.n_discard}"
    generator = deepcopy(gen_init)
    discriminator = deepcopy(disc_init)

    if args.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Train with each algorithm
    if algo in ['mean']:
        mean_est = utils.get_mean_estimator(algo)
        train.plugin_estimator_training_loop(generator, discriminator,
                                             train_dataloader, LR, LATENT_DIM,
                                             args.loss_fn, args.optim, DEVICE,
                                             args.num_iters, CHECKPOINTS,
                                             BATCH_SIZE, mean_est,
                                             N_DISCARD, args.save_dir, SUFFIX)
    elif algo == 'streaming_approx':
        train.streaming_approx_training_loop(generator, discriminator,
                                             train_dataloader, LR, LATENT_DIM,
                                             args.loss_fn, args.optim, DEVICE,
                                             args.num_iters, CHECKPOINTS,
                                             ALPHA, BATCH_SIZE, N_DISCARD,
                                             args.save_dir, SUFFIX)
