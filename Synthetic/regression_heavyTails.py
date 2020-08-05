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


def generateData(
    numSamples, numDim, Xdist, noiseDist, noiseVar, tparam, bX_pareto=10, bNoise_pareto=10
):
    if Xdist == "gaussian":
        Xrv = sp.multivariate_normal(mean=np.zeros(numDim), cov=np.eye(numDim))
        Xsamples = Xrv.rvs(numSamples)
    elif Xdist == "pareto":
        pareto_obj = sp.pareto(b=bX_pareto)
        b_mean, b_var, b_skew, b_kurt = sp.pareto.stats(b=bX_pareto, moments="mvsk")
        Xsamples = (pareto_obj.rvs(size=(numSamples, numDim)) - b_mean) / np.sqrt(b_var)

    if noiseDist == "gaussian":
        noise_rv = sp.norm(loc=0, scale=np.sqrt(noiseVar))
        noiseVec = noise_rv.rvs(numSamples)
    elif noiseDist == "pareto":
        noise_rv = sp.pareto(b=bNoise_pareto)
        b_mean, b_var, b_skew, b_kurt = sp.pareto.stats(b=bNoise_pareto, moments="mvsk")
        noiseVec = np.sqrt(noiseVar) * (noise_rv.rvs(size=numSamples) - b_mean) / np.sqrt(b_var)

    response = np.dot(Xsamples, tparam) + noiseVec

    return Xsamples, response


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def robustGD(X, y, minibatchSize, stepsSize, maxIter, tparam, confidence=0.05, meanEst="filterpd"):

    [n, p] = np.shape(X)
    theta = np.zeros(p)
    errors = np.zeros(maxIter + 1)
    errors[0] = np.linalg.norm(theta - tparam)
    rtol = np.zeros(maxIter)
    for i in range(maxIter):
        resid = np.dot(X, theta) - y
        grads = np.diag(resid) @ X

        index = np.random.permutation(np.arange(0, n))[:minibatchSize]
        grads = grads[index, :]
        if meanEst == "mean":
            meanGrad = np.mean(grads, axis=0)
        elif meanEst == "gmom":
            meanGrad = geometric_median_of_means(
                grads, int(np.ceil(3.5 * np.log(1.0 / confidence))), max_iter=100, eps=1e-5
            )
        elif meanEst == "filterpd":
            meanGrad = svd_filtering_means(
                grads, int(np.ceil(3.5 * np.log(1.0 / confidence))), "random", return_indices=False
            )

        theta_new = theta - stepsSize * meanGrad
        rtol[i] = np.linalg.norm(theta_new - theta, 2)
        theta = theta_new
        errors[i + 1] = np.linalg.norm(theta - tparam)

    return theta, errors, rtol


# Q_delta vs p
numSamples = 500
dims = [20, 40, 60, 80, 100]
Xdist = "gaussian"
noiseDist = "pareto"
noiseVar = 0.1
confidence = 0.1
numTrials = 100
stepsize = 0.1
maxIter = 400
res_ols = np.zeros((len(dims), numTrials))
res_gmom = np.zeros((len(dims), numTrials))
res_filterpd = np.zeros((len(dims), numTrials))


for j in range(len(dims)):
    numDim = dims[j]
    tparam = np.ones(numDim)
    print(numDim)
    for i in range(numTrials):
        X, y = generateData(
            numSamples, numDim, Xdist, noiseDist, noiseVar, tparam, bX_pareto=6, bNoise_pareto=3
        )

        res_ols[j, i] = np.linalg.norm(ols(X, y) - tparam, 2)
        gmom_reg, errors_gmom, rtol_gmom = robustGD(
            X, y, numSamples, stepsize, maxIter, tparam, confidence=confidence, meanEst="gmom"
        )
        res_gmom[j, i] = np.linalg.norm(gmom_reg - tparam, 2)
        filterpd_reg, errors_filterpd, rtol_filterpd = robustGD(
            X, y, numSamples, stepsize, maxIter, tparam, confidence=confidence, meanEst="filterpd"
        )
        res_filterpd[j, i] = np.linalg.norm(filterpd_reg - tparam, 2)


ols_res_combined = pd.DataFrame(res_ols).transpose().quantile(1 - confidence)
gmom_res_combined = pd.DataFrame(res_gmom).transpose().quantile(1 - confidence)
filterpd_res_combined = pd.DataFrame(res_filterpd).transpose().quantile(1 - confidence)

AllRes_plot_vs_p = dict()
AllRes_plot_vs_p["numSamples"] = numSamples
AllRes_plot_vs_p["dims"] = dims
AllRes_plot_vs_p["Xdist"] = Xdist
AllRes_plot_vs_p["noiseDist"] = noiseDist
AllRes_plot_vs_p["noiseVar"] = noiseVar
AllRes_plot_vs_p["confidence"] = confidence
AllRes_plot_vs_p["numTrials"] = numTrials
AllRes_plot_vs_p["stepsize"] = stepsize
AllRes_plot_vs_p["maxIter"] = maxIter
AllRes_plot_vs_p["ols"] = pd.DataFrame(res_ols, index=dims).transpose()
AllRes_plot_vs_p["gmom"] = pd.DataFrame(res_gmom, index=dims).transpose()
AllRes_plot_vs_p["filterpd"] = pd.DataFrame(res_filterpd, index=dims).transpose()

df = pd.DataFrame([ols_res_combined, gmom_res_combined, filterpd_res_combined]).transpose()
df.columns = ["OLS", "RGD-gmom", "RGD-filterpd"]
df.index = dims
pkl.dump(AllRes_plot_vs_p, open("Q_delta_vs_p.pkl", "wb"))

# Q_delta vs n

samples_list = np.arange(100, 600, 100)
numDim = 20
Xdist = "gaussian"
noiseDist = "pareto"
noiseVar = 0.1
confidence = 0.1
numTrials = 100
stepsize = 0.1
maxIter = 400

res_ols = np.zeros((len(samples_list), numTrials))
res_gmom = np.zeros((len(samples_list), numTrials))
res_filterpd = np.zeros((len(samples_list), numTrials))

for j in range(len(samples_list)):
    numSamples = samples_list[j]
    tparam = np.ones(numDim)
    print(numDim)
    print(numSamples)
    for i in range(numTrials):
        X, y = generateData(
            numSamples, numDim, Xdist, noiseDist, noiseVar, tparam, bX_pareto=6, bNoise_pareto=3
        )

        res_ols[j, i] = np.linalg.norm(ols(X, y) - tparam, 2)
        gmom_reg, errors_gmom, rtol_gmom = robustGD(
            X, y, numSamples, stepsize, maxIter, tparam, confidence=confidence, meanEst="gmom"
        )
        res_gmom[j, i] = np.linalg.norm(gmom_reg - tparam, 2)
        filterpd_reg, errors_filterpd, rtol_filterpd = robustGD(
            X, y, numSamples, stepsize, maxIter, tparam, confidence=confidence, meanEst="filterpd"
        )
        res_filterpd[j, i] = np.linalg.norm(filterpd_reg - tparam, 2)


ols_res_combined = pd.DataFrame(res_ols).transpose().quantile(1 - confidence)
gmom_res_combined = pd.DataFrame(res_gmom).transpose().quantile(1 - confidence)
filterpd_res_combined = pd.DataFrame(res_filterpd).transpose().quantile(1 - confidence)

AllRes_plot_vs_n = dict()
AllRes_plot_vs_n["numSamples"] = samples_list
AllRes_plot_vs_n["dim"] = numDim
AllRes_plot_vs_n["Xdist"] = Xdist
AllRes_plot_vs_n["noiseDist"] = noiseDist
AllRes_plot_vs_n["noiseVar"] = noiseVar
AllRes_plot_vs_n["confidence"] = confidence
AllRes_plot_vs_n["numTrials"] = numTrials
AllRes_plot_vs_n["stepsize"] = stepsize
AllRes_plot_vs_n["maxIter"] = maxIter
AllRes_plot_vs_n["ols"] = pd.DataFrame(res_ols, index=samples_list).transpose()
AllRes_plot_vs_n["gmom"] = pd.DataFrame(res_gmom, index=samples_list).transpose()
AllRes_plot_vs_n["filterpd"] = pd.DataFrame(res_filterpd, index=samples_list).transpose()


df = pd.DataFrame(
    [ols_res_combined, gmom_res_combined, filterpd_res_combined]
).transpose()  # .plot()
df.columns = ["OLS", "RGD-gmom", "RGD-filterpd"]
df.index = samples_list
pkl.dump(AllRes_plot_vs_n, open("Q_delta_vs_n.pkl", "wb"))

# Q_delta vs delta

confidence_List = np.linspace((0.01), (0.1), 5)
numDim = 20
numSamples = 500
Xdist = "gaussian"
noiseDist = "pareto"
noiseVar = 0.1
numTrials = 100
stepSize = 0.1
maxIter = 400
tparam = np.ones(numDim)
res_ols = np.zeros((len(confidence_List), numTrials))
res_gmom = np.zeros((len(confidence_List), numTrials))
res_filterpd = np.zeros((len(confidence_List), numTrials))


for j in range(len(confidence_List)):
    confidence = confidence_List[j]
    for i in range(numTrials):
        X, y = generateData(
            numSamples, numDim, Xdist, noiseDist, noiseVar, tparam, bX_pareto=6, bNoise_pareto=3
        )

        res_ols[j, i] = np.linalg.norm(ols(X, y) - tparam, 2)
        gmom_reg, _, _ = robustGD(
            X, y, numSamples, stepsize, maxIter, tparam, confidence=confidence, meanEst="gmom"
        )
        res_gmom[j, i] = np.linalg.norm(gmom_reg - tparam, 2)
        filterpd_reg, _, _ = robustGD(
            X, y, numSamples, stepsize, maxIter, tparam, confidence=confidence, meanEst="filterpd"
        )
        res_filterpd[j, i] = np.linalg.norm(filterpd_reg - tparam, 2)

confidence_list = confidence_List
ols_res_combined = [
    pd.Series(pd.DataFrame(res_ols).values.flatten()).quantile(1 - confidence_list[i])
    for i in range(len(confidence_list))
]
gmom_res_combined = [
    pd.DataFrame(res_gmom).transpose()[i].quantile(1 - confidence_list[i])
    for i in range(len(confidence_list))
]

filterpd_res_combined = [
    pd.DataFrame(res_filterpd).transpose()[i].quantile(1 - confidence_list[i])
    for i in range(len(confidence_list))
]

AllRes_plot_vs_delta = dict()
AllRes_plot_vs_delta["numSamples"] = numSamples
AllRes_plot_vs_delta["dim"] = numDim
AllRes_plot_vs_delta["Xdist"] = Xdist
AllRes_plot_vs_delta["noiseDist"] = noiseDist
AllRes_plot_vs_delta["noiseVar"] = noiseVar
AllRes_plot_vs_delta["confidence"] = confidence_List
AllRes_plot_vs_delta["numTrials"] = numTrials
AllRes_plot_vs_delta["stepsize"] = stepsize
AllRes_plot_vs_delta["maxIter"] = maxIter
AllRes_plot_vs_delta["ols"] = pd.DataFrame(res_ols, index=confidence_List).transpose()
AllRes_plot_vs_delta["gmom"] = pd.DataFrame(res_gmom, index=confidence_List).transpose()
AllRes_plot_vs_delta["filterpd"] = pd.DataFrame(res_filterpd, index=confidence_List).transpose()

df = pd.DataFrame(
    [ols_res_combined, gmom_res_combined, filterpd_res_combined]
).transpose()  # .plot()
df.columns = ["OLS", "RGD-gmom", "RGD-filterpd"]
df.index = np.sqrt(np.log(1.0 / confidence_List))
pkl.dump(AllRes_plot_vs_delta, open("Q_delta_vs_delta.pkl"))
