import numpy as np
from scipy.sparse.linalg import svds


def filtering_means(samples, n_discard, discard_mode,
                    return_indices=False):
    """
    Compute the filtering estimator for samples by computing
    an approximate leading eigenvector.
    """
    if discard_mode not in ['greedy', 'random']:
        raise ValueError("Invalid argument to discard_mode")

    if len(samples.shape) > 1:
        num_values, dim = samples.shape
    else:
        num_values, dim = len(samples), 1

    excl_indices = []
    all_indices = np.arange(num_values)
    for _ in range(n_discard):
        sample_mean = np.mean(samples, axis=0)

        if dim > 1:
            v = svds((samples - sample_mean) / np.sqrt(num_values),
                     k=1)[2].squeeze()
            v = v.squeeze()
            z = ((samples - sample_mean) @ v) ** 2
        else:
            v = np.ones(1)
            z = (samples - sample_mean) ** 2

        if discard_mode == 'greedy':
            index = np.argmax(z)
            samples = np.delete(samples, index, axis=0)
            excl_indices.append(all_indices[index].item())
            all_indices = np.delete(all_indices, index)
            num_values -= 1
        else:
            index = (1 - np.random.multinomial(1, z / np.sum(z))).astype(bool)
            samples = samples[index]
            excl_indices.append(all_indices[~index].item())
            all_indices = all_indices[index]
            num_values -= 1

    if return_indices:
        return np.mean(samples, axis=0), np.array(excl_indices)
    else:
        return np.mean(samples, axis=0)


def streaming_update_algorithm(samples, n_discard, discard_mode='greedy',
                               top_v=None, top_lambda=None, old_mean=None, alpha=0.5):
    """
    If top_v and top_lambda are given, then use them to update the eigenvector
    with a moving average, whose factor is alpha
    """
    n = samples.shape[0]
    if top_v is None:
        old_mean = np.mean(samples, axis=0)
        centered_samples = samples - old_mean
        output = svds(centered_samples, k=1, return_singular_vectors="vh")
        top_v = output[2].squeeze()
        top_lambda = output[1][0]
    else:
        old_mean = (1 - alpha) * old_mean + alpha * np.mean(samples, axis=0)
        centered_samples = samples - old_mean
        n = centered_samples.shape[0]
        b_t = np.hstack([np.sqrt((1 - alpha) * top_lambda) * top_v.reshape(-1, 1),
                         centered_samples.T * np.sqrt(alpha / (n - 1))])  # Size d x (n + 1)
        output = svds(b_t, k=1, return_singular_vectors="vh")
        top_v = b_t @ output[2].squeeze()
        top_v /= np.linalg.norm(top_v)
        top_lambda = output[1][0]

    z = (centered_samples @ top_v) ** 2
    if discard_mode == 'greedy':
        index = np.argpartition(z, n - n_discard)[:n - n_discard]
    else:
        excl_index = np.random.choice(len(samples), size=n_discard, replace=False, p=z / z.sum())
        index = np.ones(n, dtype=bool)
        index[excl_index] = False

    return np.mean(samples[index], axis=0), top_v, top_lambda, old_mean
