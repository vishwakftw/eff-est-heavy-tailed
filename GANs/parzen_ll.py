# Code adapted from https://github.com/goodfeli/adversarial/blob/master/parzen_ll.py
import argparse
import time
import gc
import numpy
import theano
import theano.tensor as T


def get_nll(x, parzen, batch_size=10):
    """
    Credit: Yann N. Dauphin
    """

    inds = range(x.shape[0])
    n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

    times = []
    nlls = []
    for i in range(n_batches):
        begin = time.time()
        samps = x[inds[i::n_batches]]
        nll = parzen(samps)
        end = time.time()
        times.append(end-begin)
        nlls.extend(nll)

        if i % 10 == 0:
            print(i, numpy.mean(times), numpy.mean(nlls))

    return numpy.array(nlls)


def log_mean_exp(a):
    """
    Credit: Yann N. Dauphin
    """

    max_ = a.max(1)

    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def theano_parzen(mu, sigma):
    """
    Credit: Yann N. Dauphin
    """
    x = T.matrix()
    mu = theano.shared(mu)
    a = (x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1)) / sigma
    E = log_mean_exp(-0.5 * (a ** 2).sum(2))
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))

    return theano.function([x], E - Z)


def cross_validate_sigma(samples, data, sigmas, batch_size):

    lls = []
    for sigma in sigmas:
        parzen = theano_parzen(samples, sigma)
        tmp = get_nll(data, parzen, batch_size=batch_size)
        lls.append(numpy.asarray(tmp).mean())
        del parzen
        gc.collect()

    ind = numpy.argmax(lls)
    return sigmas[ind]


def get_valid(ds):
    data = numpy.load('dataset/mnist_train.npy')[50000:60000]
    return numpy.matrix(data)


def get_test(ds):
    data = numpy.load('dataset/mnist_test.npy')
    return numpy.matrix(data)


def main():
    parser = argparse.ArgumentParser(description='Parzen window, log-likelihood estimator')
    parser.add_argument('-s', '--sigma', default=None)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-c', '--cross_val', default=10, type=int,
                        help="Number of cross validation folds")
    parser.add_argument('--sigma_start', default=-1, type=float)
    parser.add_argument('--sigma_end', default=0., type=float)
    parser.add_argument('--imgs_loc', required=True)
    args = parser.parse_args()

    # generate samples
    samples = numpy.load(args.imgs_loc)
    samples = (samples + 1) / 2  # bring to [0, 1]

    # cross validate sigma
    if args.sigma is None:
        valid = get_valid()
        sigma_range = numpy.logspace(args.sigma_start, args.sigma_end, num=args.cross_val)
        sigma = cross_validate_sigma(samples, valid, sigma_range, args.batch_size)
    else:
        sigma = float(args.sigma)

    print("Using Sigma: {}".format(sigma))
    gc.collect()

    # fit and evaulate
    parzen = theano_parzen(samples, sigma)
    ll = get_nll(get_test(), parzen, batch_size=args.batch_size)

    print("Log-Likelihood of test set = {}".format(ll.mean()))


if __name__ == "__main__":
    main()
