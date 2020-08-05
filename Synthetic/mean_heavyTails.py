#!/usr/bin/env python
import numpy as np

import pandas as pd
import pickle as pkl

import scipy.stats as sp
import scipy.sparse.linalg as ssl


def svd_filtering_means(samples, n_discard, discard_mode, return_indices=False):
    """
    Compute the filtering estimator for samples by computing
    an approximate leading eigenvector.
    """
    if discard_mode not in ["greedy", "random"]:
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
            v = ssl.svds((samples - sample_mean) / np.sqrt(num_values), k=1)[2].squeeze()
            v = v.squeeze()
            z = ((samples - sample_mean) @ v) ** 2
        else:
            v = np.ones(1)
            z = (samples - sample_mean) ** 2

        if discard_mode == "greedy":
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


def geometric_median_of_means(samples, num_buckets, max_iter=100, eps=1e-5):
    """
    Compute the geometric median of means by placing `samples`
    in num_buckets using Weiszfeld's algorithm
    """
    if len(samples.shape) == 1:
        samples = samples.reshape(-1, 1)
    bucketed_means = np.array(
        [np.mean(val, axis=0) for val in np.array_split(samples, num_buckets)]
    )

    if bucketed_means.shape[0] == 1:
        return bucketed_means.squeeze()  # when sample size is 1, the only sample is the median

    # This reduces the chance that the initial estimate is close to any
    # one of the data points
    gmom_est = np.mean(bucketed_means, axis=0)

    for i in range(max_iter):
        weights = np.reciprocal(np.linalg.norm(bucketed_means - gmom_est, axis=1, ord=2))[
            :, np.newaxis
        ]
        old_gmom_est = gmom_est
        gmom_est = np.sum(bucketed_means * weights, axis=0) / np.sum(weights, axis=0)
        if (
            np.linalg.norm(gmom_est - old_gmom_est, ord=2) / np.linalg.norm(old_gmom_est, ord=2)
            < eps
        ):
            break

    return gmom_est


def generateData_mean(numSamples, numDim, Xdist, bX_pareto=10):
    pareto_obj = sp.pareto(b=bX_pareto)
    b_mean, b_var, b_skew, b_kurt = sp.pareto.stats(b=bX_pareto, moments="mvsk")
    Xsamples = (pareto_obj.rvs(size=(numSamples, numDim)) - b_mean) / np.sqrt(b_var)
    return Xsamples


def samplemean(X):
    return np.mean(X, axis=0)


# Q_delta vs p
numSamples = 500
dims = [20, 40, 60, 80, 100]
confidence = 0.05
numTrials = 2000

res_mean = np.zeros((len(dims), numTrials))
res_gmom = np.zeros((len(dims), numTrials))
res_filterpd = np.zeros((len(dims), numTrials))

for j in range(len(dims)):
    numDim = dims[j]
    tparam = np.zeros(numDim)
    for i in range(numTrials):
        X = generateData_mean(numSamples, numDim, bX_pareto=3)
        res_mean[j, i] = np.linalg.norm(samplemean(X) - tparam, 2)
        gmom_reg = geometric_median_of_means(
            X, int(np.ceil(3.5 * np.log(1.0 / confidence))), max_iter=100, eps=1e-5
        )
        res_gmom[j, i] = np.linalg.norm(gmom_reg - tparam, 2)
        filterpd_reg = svd_filtering_means(
            X, int(np.ceil(3.5 * np.log(1.0 / confidence))), "greedy", return_indices=False
        )

        res_filterpd[j, i] = np.linalg.norm(filterpd_reg - tparam, 2)

mean_res_combined = pd.DataFrame(res_mean).transpose().quantile(1 - confidence)
gmom_res_combined = pd.DataFrame(res_gmom).transpose().quantile(1 - confidence)
filterpd_res_combined = pd.DataFrame(res_filterpd).transpose().quantile(1 - confidence)

df = pd.DataFrame([mean_res_combined, gmom_res_combined, filterpd_res_combined]).transpose()
df.columns = ["mean", "gmom", "filterpd"]
df.index = dims

pkl.dump(df, open("Q_delta_vs_p.pkl", "wb"))

samples = [100, 200, 300, 400, 500]
numDim = 50
confidence = 0.05
numTrials = 2000

res_mean = np.zeros((len(samples), numTrials))
res_gmom = np.zeros((len(samples), numTrials))
res_filterpd = np.zeros((len(samples), numTrials))

for j in range(len(samples)):
    numSamples = samples[j]
    tparam = np.zeros(numDim)
    for i in range(numTrials):
        X = generateData_mean(numSamples, numDim, bX_pareto=3)
        res_mean[j, i] = np.linalg.norm(samplemean(X) - tparam, 2)
        gmom_reg = geometric_median_of_means(
            X, int(np.ceil(3.5 * np.log(1.0 / confidence))), max_iter=100, eps=1e-5
        )
        res_gmom[j, i] = np.linalg.norm(gmom_reg - tparam, 2)
        filterpd_reg = svd_filtering_means(
            X, int(np.ceil(3.5 * np.log(1.0 / confidence))), "greedy", return_indices=False
        )

        res_filterpd[j, i] = np.linalg.norm(filterpd_reg - tparam, 2)

mean_res_combined = pd.DataFrame(res_mean).transpose().quantile(1 - confidence)
gmom_res_combined = pd.DataFrame(res_gmom).transpose().quantile(1 - confidence)
filterpd_res_combined = pd.DataFrame(res_filterpd).transpose().quantile(1 - confidence)

df = pd.DataFrame([mean_res_combined, gmom_res_combined, filterpd_res_combined]).transpose()
df.columns = ["mean", "gmom", "filterpd"]
df.index = samples

pkl.dump(df, open("Q_delta_vs_n.pkl", "wb"))

numSamples = 300
numDim = 50
confidences = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.9, 0.1]
numTrials = 2000

res_mean = np.zeros((len(confidences), numTrials))
res_gmom = np.zeros((len(confidences), numTrials))
res_filterpd = np.zeros((len(confidences), numTrials))

for j in range(len(samples)):
    confidence = confidences[j]
    tparam = np.zeros(numDim)
    for i in range(numTrials):
        X = generateData_mean(numSamples, numDim, bX_pareto=3)
        res_mean[j, i] = np.linalg.norm(samplemean(X) - tparam, 2)
        gmom_reg = geometric_median_of_means(
            X, int(np.ceil(3.5 * np.log(1.0 / confidence))), max_iter=100, eps=1e-5
        )
        res_gmom[j, i] = np.linalg.norm(gmom_reg - tparam, 2)
        filterpd_reg = svd_filtering_means(
            X, int(np.ceil(3.5 * np.log(1.0 / confidence))), "greedy", return_indices=False
        )

        res_filterpd[j, i] = np.linalg.norm(filterpd_reg - tparam, 2)

mean_res_combined = [np.quantile(res_mean[i], 1 - confidences[i]) for i in range(0, 10)]
gmom_res_combined = [np.quantile(res_gmom[i], 1 - confidences[i]) for i in range(0, 10)]
filterpd_res_combined = [np.quantile(res_filterpd[i], 1 - confidences[i]) for i in range(0, 10)]

df = pd.DataFrame([mean_res_combined, gmom_res_combined, filterpd_res_combined]).transpose()
df.columns = ["mean", "gmom", "filterpd"]
df.index = confidences

pkl.dump(df, open("Q_delta_vs_delta.pkl", "wb"))
