### Code for experiments in Robust Estimation via Robust Gradient Estimation and Efficient Estimators for Heavy-Tailed Machine Learning

##### Code Structure
There are 4 folders, each of which are described below:
- `GANs`: This folder contains scripts for experiments pertaining to [Deep Convolutional GANs](https://arxiv.org/abs/1511.06434) trained on the MNIST and CIFAR10 datasets.
- `RealNVP`: This folder contains scripts for experiments pertaining to the [Real-NVP](https://arxiv.org/abs/1605.08803) normalizing flow model trained on the CIFAR10 dataset.
- `Synthetic`: This folder contains scripts for synthetic experiments for mean estimation and heavy-tailed linear regression.
- `misc`: This contains code for miscellaneous purposes such as the `alpha`-index estimator.

##### Running the code
Dependencies include:
- PyTorch
- SciPy
- NumPy
- Pandas
- Matplotlib
- Theano (for Parzen-window estimates)

For experiments pertaining to GANs and RealNVP, enter the respective folders and run:
```bash
python main.py [OPTIONS]
```

To view a list of available options with their descriptions, use:
```bash
python main.py --help
```

For synthetic experiments, enter `Synthetic/` and run:
```
python mean_heavyTails.py
python regression_heavyTails.py
```

##### Finding the results
For the GAN and RealNVP experiments, the models are saved in the directory specified by `--save_dir`. To compute metric, you are required to load the models and obtain metrics separately.

For the synthetic experiments, the results are saved in pickle files in `Synthetic/`.

##### Acknowledgements
- The code for Parzen-window based log-likelihood estimates is adapted from the original [GAN source code by Ian Goodfellow](https://github.com/goodfeli/adversarial).
- The implementation of Real-NVP is adapted from [Chris Chute's implementation](https://github.com/chrischute/real-nvp).
- The code for computing the Inception and MODE scores is based on [Shane Barratt's implementation](https://github.com/sbarratt/inception-score-pytorch).
- For the MODE score, a high-quality classifier is required to be trained on the MNIST dataset - the code for this is taken from [pytorch/examples](https://github.com/pytorch/examples/tree/master/mnist).
- The code for the `alpha`-index estimator is borrowed from [Umut Simsekli's implementation](https://github.com/umutsimsekli/sgd_tail_index).

The respective copyright notices are attached in the respective folders.

##### References

[1] Adarsh Prasad, Arun Sai Suggala, Sivaraman Balakrishnan and Pradeep Ravikumar, _Robust estimation via robust gradient estimation_, Journal of the Royal Statistical Society Series B, 2020

[2] Adarsh Prasad\*, Vishwak Srinivasan\*, Sivaraman Balakrishnan and Pradeep Ravikumar, _Efficient Estimators for Heavy-Tailed Machine Learning_, under submission, 2020
