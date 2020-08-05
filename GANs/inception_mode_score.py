# Code adapted from https://github.com/sbarratt/inception-score-pytorch
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

import utils
from functools import partial


def get_prediction_scores(x, resize, model, up=None):
    if resize and up is None:
        raise ValueError("can't resize without upsampling")

    with torch.no_grad():
        out = x.clone()
        if resize:
            out = up(out)
        out = model(out)
    return F.softmax(out, 1).cpu().numpy()


def inception_score(imgs, device='cpu', batch_size=32, resize=False, splits=1,
                    inception_model=None):
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    if inception_model is None:
        inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        inception_model.eval()

    if resize:
        up = torch.nn.Upsample(size=(299, 299), mode='bilinear').to(device)
    else:
        up = None

    # Get predictions
    preds = []
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        preds.append(get_prediction_scores(batch, resize=resize, model=inception_model, up=up))
    preds = np.vstack(preds)

    # Now compute the mean kl-div
    split_scores = []
    for part in np.array_split(preds, splits):
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def mode_score(imgs, classifier_model, device='cpu', batch_size=32, splits=1):
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Get predictions
    preds = []
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        preds.append(get_prediction_scores(batch, resize=False, model=classifier_model, up=None))
    preds = np.vstack(preds)

    # Now compute the mean kl-div
    split_scores = []
    for part in np.array_split(preds, splits):
        py = np.mean(part, axis=0)
        py_dataset = np.full_like(py, 0.1)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py_dataset))
        split_scores.append(np.exp(np.mean(scores) - entropy(py, py_dataset)))

    return np.mean(split_scores), np.std(split_scores)


def compute_score(model, dataset, num_imgs=10000, pretrained_model_loc=None, device='cpu'):
    gen_cons, _ = utils.get_model_constructors(dataset)

    if dataset == 'mnist':
        if pretrained_model_loc is None:
            raise ValueError("Need to specify the location of the MNIST classifier")

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = F.log_softmax(x, dim=1)
                return output

        classifier_model = Net()
        classifier_model = classifier_model.to(device=device)
        classifier_model.load_state_dict(torch.load(pretrained_model_loc, map_location=device))
        classifier_model.eval()

        scoring_function = partial(mode_score, classifier_model=classifier_model,
                                   device=device, splits=10, batch_size=100)

    elif dataset == 'cifar10':
        inception_model = inception_v3(pretrained=True, transform_input=False)
        inception_model = inception_model.to(device)
        inception_model.eval()

        scoring_function = partial(inception_score, inception_model=inception_model,
                                   device=device, splits=10, batch_size=100, resize=True)

    generator = gen_cons(128)
    imgs = []
    for i in range(10):
        imgs.append(generator(torch.randn(1000, 128, device='cuda')).detach().cpu())
    imgs = torch.cat(imgs, dim=0)
    score = scoring_function(imgs=imgs)[0]
    return score
