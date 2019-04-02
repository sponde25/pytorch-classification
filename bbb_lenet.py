import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print(torch.cuda.is_available())
out = 'result'

BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
data = Dataset('mnist')
train_loader = data.get_train_loader(BATCH_SIZE)
test_loader = data.get_test_loader(TEST_BATCH_SIZE)

TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(test_loader)

CLASSES = 10
TRAIN_EPOCHS = 300
SAMPLES = 2
TEST_SAMPLES = 10

PI = 0.5
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, prune=None):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        if prune is not None:
            snr = torch.abs(self.mu).div(self.sigma)
            mask = snr >= prune
            sample = self.mu + self.sigma * epsilon
            sample = sample.mul(mask.float().to(sample.device))
            return sample
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False, prune=None):

        if self.training or sample:
            weight = self.weight.sample(prune)
            bias = self.bias.sample(prune)
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianConvolutional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False, prune=None):

        if self.training or sample:
            weight = self.weight.sample(prune)
            bias = self.bias.sample(prune)
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.conv2d(input, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLinear(28 * 28, 1200)
        self.l2 = BayesianLinear(1200, 1200)
        self.l3 = BayesianLinear(1200, 10)

    def forward(self, x, sample=False, prune=None):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample, prune=prune))
        x = F.relu(self.l2(x, sample, prune=prune))
        x = self.l3(x, sample, prune=prune)
        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior

    def sample_elbo(self, input, target, samples=SAMPLES):
        BATCH_SIZE = input.shape[0]
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.cross_entropy(outputs.mean(0), target, reduction='sum')
        loss = (log_variational_posterior - log_prior) / NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


class BayesianLeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.cl1 = BayesianConvolutional(1, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.cl2 = BayesianConvolutional(6, 16, 5)
        self.fc1 = BayesianLinear(16 * 4 * 4, 120)
        self.fc2 = BayesianLinear(120, 84)
        self.fc = BayesianLinear(84, 10)

    def forward(self, x, sample=False, prune=None):
        x = self.pool(F.relu(self.cl1(x, sample, prune=prune)))
        x = self.pool(F.relu(self.cl2(x, sample, prune=prune)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x, sample, prune=prune))
        x = F.relu(self.fc2(x, sample, prune=prune))
        x = self.fc(x, sample, prune=prune)
        return x

    def log_prior(self):
        return self.cl1.log_prior \
               + self.cl2.log_prior \
               + self.fc1.log_prior \
               + self.fc2.log_prior \
               + self.fc.log_prior

    def log_variational_posterior(self):
        return self.cl1.log_variational_posterior \
               + self.cl2.log_variational_posterior \
               + self.fc1.log_variational_posterior \
               + self.fc2.log_variational_posterior \
               + self.fc.log_variational_posterior

    def sample_elbo(self, input, target, samples=SAMPLES):
        BATCH_SIZE = input.shape[0]
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.cross_entropy(outputs.mean(0), target, reduction='sum')
        loss = (log_variational_posterior - log_prior) / NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


def train(net, optimizer, epoch):
    net.train()
    total_loss = 0.
    ce_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        ce_loss += negative_log_likelihood.detach().item()
    print('Loss: {}'.format(ce_loss / TRAIN_SIZE))


def softmax_predictive_accuracy(logits_list, y, ret_loss=False):
    probs_list = [F.log_softmax(logits, dim=1) for logits in logits_list]
    probs_tensor = torch.stack(probs_list, dim=2)
    probs = torch.mean(probs_tensor, dim=2)
    if ret_loss:
        loss = F.nll_loss(probs, y, reduction='sum').item()
    _, pred_class = torch.max(probs, 1)
    correct = pred_class.eq(y.view_as(pred_class)).sum().item()
    if ret_loss:
        return correct, loss
    return correct


def test_ensemble():
    net.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = []
            for i in range(TEST_SAMPLES):
                outputs.append(net(data, sample=True))
            _c, _l = softmax_predictive_accuracy(outputs, target, ret_loss=True)
            correct += _c
            loss += _l
    return correct, loss / TEST_SIZE

net = BayesianLeNet5().to(DEVICE)
optimizer = optim.Adam(net.parameters())
test_correct, test_loss = [], []
best_correct = 0.
for epoch in range(TRAIN_EPOCHS):
    train(net, optimizer, epoch)
    _c, _l = test_ensemble()
    test_correct.append(_c)
    test_loss.append(_l)
    print('Test Correct: {}/{}, Test Loss: {}'.format(_c, TEST_SIZE, _l))
    if best_correct < _c:
        torch.save(net.state_dict(), 'lenet5_mnist.pt')

torch.save(test_correct, 'lenet5_mnist_correct.arr')
torch.save(test_loss, 'lenet5_mnist_loss.arr')