import torch

import utils
import util
from models import RealNVPLoss
from estimators import streaming_update_algorithm


def streaming_approx_training_loop(real_nvp_model, dataloader,
                                   learning_rate, optim, device,
                                   total_iters, checkpoint_intervals,
                                   alpha, batchsize, n_discard,
                                   weight_decay, max_grad_norm,
                                   save_dir, save_suffix):
    """
    Function to train the RealNVP model using
    the streaming rank-1 approximation with algorithm for total_iters
    with optimizer optim
    """
    param_groups = util.get_param_groups(real_nvp_model, weight_decay, norm_suffix='weight_g')
    optimizer_cons = utils.get_optimizer_cons(optim, learning_rate)
    optimizer = optimizer_cons(param_groups)

    loss_fn = RealNVPLoss()
    flag = False
    iteration = 0
    top_eigvec, top_eigval, running_mean = None, None, None

    real_nvp_model.train()
    while not flag:
        for x, _ in dataloader:
            # Update iteration counter
            iteration += 1

            x = x.to(device)
            z, sldj = real_nvp_model(x, reverse=False)
            unaggregated_loss = loss_fn(z, sldj, aggregate=False)
            # First sample gradients
            sgradients = utils.gradient_sampler(unaggregated_loss,
                                                real_nvp_model)
            # Then get the estimate with the previously computed direction
            stoc_grad, top_eigvec, top_eigval, running_mean = streaming_update_algorithm(sgradients,
                                                                                         n_discard=n_discard,
                                                                                         top_v=top_eigvec,
                                                                                         top_lambda=top_eigval,
                                                                                         old_mean=running_mean,
                                                                                         alpha=alpha)
            # Perform the update of .grad attributes
            with torch.no_grad():
                utils.update_grad_attributes(real_nvp_model.parameters(),
                                             torch.as_tensor(stoc_grad, device=device))
            # Clip gradient if required
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            # Perform the update
            optimizer.step()

            if iteration in checkpoint_intervals:
                print(f"Completed {iteration}")
                torch.save(real_nvp_model.state_dict(),
                           f"{save_dir}/real_nvp_streaming_approx_{iteration}_{save_suffix}.pt")

            if iteration == total_iters:
                flag = True
                break

    return real_nvp_model


def plugin_estimator_training_loop(real_nvp_model, dataloader,
                                   learning_rate, optim, device,
                                   total_iters, checkpoint_intervals,
                                   batchsize, algorithm, weight_decay,
                                   max_grad_norm, save_dir, save_suffix):
    """
    Function to train the RealNVP model using
    a plugin mean estimation algorithm for total_iters with learning_rate
    """
    param_groups = util.get_param_groups(real_nvp_model, weight_decay, norm_suffix='weight_g')
    optimizer_cons = utils.get_optimizer_cons(optim, learning_rate)
    optimizer = optimizer_cons(param_groups)

    loss_fn = RealNVPLoss()
    flag = False
    iteration = 0
    while not flag:
        for x, _ in dataloader:
            # Update iteration counter
            iteration += 1

            x = x.to(device)
            z, sldj = real_nvp_model(x, reverse=False)
            unaggregated_loss = loss_fn(z, sldj, aggregate=False)
            if algorithm.__name__ == 'mean':
                agg_loss = unaggregated_loss.mean()
                agg_loss.backward()
            else:
                # First sample gradients
                sgradients = utils.gradient_sampler(unaggregated_loss,
                                                    real_nvp_model)
                # Then get the estimate with the mean estimation algorithm
                stoc_grad = algorithm(sgradients)
                # Perform the update of .grad attributes
                with torch.no_grad():
                    utils.update_grad_attributes(real_nvp_model.parameters(),
                                                 torch.as_tensor(stoc_grad, device=device))
            # Clip gradient if required
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            # Perform the update
            optimizer.step()

            if iteration in checkpoint_intervals:
                print(f"Completed {iteration}")
                torch.save(real_nvp_model.state_dict(),
                           f"{save_dir}/real_nvp_{algorithm.__name__}_{iteration}_{save_suffix}.pt")

            if iteration == total_iters:
                flag = True
                break

    return real_nvp_model
