import torch
import utils
import losses
from estimators import streaming_update_algorithm


def streaming_approx_training_loop(generator, discriminator,
                                   dataloader, learning_rate, latent_dim,
                                   loss_function, optim, device,
                                   total_iters, checkpoint_intervals,
                                   alpha, batchsize, n_discard,
                                   save_dir, save_suffix):
    """
    Function to train a generator and discriminator for a GAN using
    the streaming rank-1 approximation with algorithm for total_iters
    with optimizer optim
    """
    optimizer_cons = utils.get_optimizer_cons(optim, learning_rate)

    disc_optimizer = optimizer_cons(discriminator.parameters())
    gen_optimizer = optimizer_cons(generator.parameters())

    discriminator_loss, generator_loss = losses.get_loss_fn(loss_function)
    flag = False
    iteration = 0
    top_eigvec, top_eigval, running_mean = None, None, None

    while not flag:
        for real_image_batch, _ in dataloader:
            # Update iteration counter
            iteration += 1

            # Discriminator: standard training
            fake_image_batch = generator(torch.randn(real_image_batch.shape[0],
                                                     latent_dim, device=device))
            real_pred = discriminator(real_image_batch.to(device))
            fake_pred = discriminator(fake_image_batch.detach())
            real_loss, fake_loss = discriminator_loss(real_pred, fake_pred)
            disc_loss = torch.mean(real_loss + fake_loss)
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            # Generator: proper mean estimation
            # First sample gradients
            sgradients = utils.gradient_sampler(discriminator, generator,
                                                generator_loss,
                                                noise_batchsize=batchsize,
                                                latent_dim=latent_dim,
                                                device=device)
            # Then get the estimate with the previously computed direction
            stoc_grad, top_eigvec, top_eigval, running_mean = streaming_update_algorithm(sgradients,
                                                                                         n_discard=n_discard(iteration),
                                                                                         top_v=top_eigvec,
                                                                                         top_lambda=top_eigval,
                                                                                         old_mean=running_mean,
                                                                                         alpha=alpha)
            # Perform the update of .grad attributes
            with torch.no_grad():
                utils.update_grad_attributes(generator.parameters(),
                                             torch.as_tensor(stoc_grad, device=device))
            # Perform the update
            gen_optimizer.step()

            if iteration in checkpoint_intervals:
                print(f"Completed {iteration}")
                torch.save(generator.state_dict(),
                           f"{save_dir}/generator_streaming_approx_{iteration}_{save_suffix}.pt")
                torch.save(discriminator.state_dict(),
                           f"{save_dir}/discriminator_streaming_approx_{iteration}_{save_suffix}.pt")

            if iteration == total_iters:
                flag = True
                break

    return generator, discriminator


def plugin_estimator_training_loop(generator, discriminator,
                                   dataloader, learning_rate, latent_dim,
                                   loss_function, optim, device,
                                   total_iters, checkpoint_intervals,
                                   batchsize, algorithm, n_discard,
                                   save_dir, save_suffix):
    """
    Function to train a generator and discriminator for a GAN using
    a plugin mean estimation algorithm for total_iters with learning_rate
    """
    optimizer_cons = utils.get_optimizer_cons(optim, learning_rate)

    disc_optimizer = optimizer_cons(discriminator.parameters())
    gen_optimizer = optimizer_cons(generator.parameters())

    discriminator_loss, generator_loss = losses.get_loss_fn(loss_function)
    flag = False
    iteration = 0
    while not flag:
        for real_image_batch, _ in dataloader:
            # Update iteration counter
            iteration += 1

            # Discriminator: standard training
            fake_image_batch = generator(torch.randn(real_image_batch.shape[0],
                                                     latent_dim, device=device))
            real_pred = discriminator(real_image_batch.to(device))
            fake_pred = discriminator(fake_image_batch.detach())
            real_loss, fake_loss = discriminator_loss(real_pred, fake_pred)
            disc_loss = torch.mean(real_loss + fake_loss)
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            if algorithm.__name__ == 'mean':
                fake_images = generator(torch.randn(batchsize, latent_dim, device=device))
                fake_preds = discriminator(fake_images).squeeze()
                gen_loss = generator_loss(fake_preds).mean()
                gen_optimizer.zero_grad()
                gen_loss.backward()
            else:
                # Generator: proper mean estimation
                # First sample gradients
                sgradients = utils.gradient_sampler(discriminator, generator,
                                                    generator_loss,
                                                    noise_batchsize=batchsize,
                                                    latent_dim=latent_dim,
                                                    device=device)
                # Then get the estimate with the mean estimation algorithm
                stoc_grad = algorithm(sgradients, n_discard=n_discard(iteration))
                # Perform the update of .grad attributes
                with torch.no_grad():
                    utils.update_grad_attributes(generator.parameters(),
                                                 torch.as_tensor(stoc_grad, device=device))
            # Perform the update
            gen_optimizer.step()

            if iteration in checkpoint_intervals:
                print(f"Completed {iteration}")
                torch.save(generator.state_dict(),
                           f"{save_dir}/generator_{algorithm.__name__}_{iteration}_{save_suffix}.pt")
                torch.save(discriminator.state_dict(),
                           f"{save_dir}/discriminator_{algorithm.__name__}_{iteration}_{save_suffix}.pt")

            if iteration == total_iters:
                flag = True
                break

    return generator, discriminator
