import os
import argparse
from importlib import import_module
import shutil
import numpy as np

import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import models.cifar as models
from datasets import Dataset
from vogn import VOGN

OPTIMIZER_VOGN = 'vogn'
OPTIMIZER_ADAM = 'adam'
OPTIMIZER_SGD = 'sgd'

LR_SCHEDULE_CONSTANT = 'constant'
LR_SCHEDULE_EXPONENTIAL = 'exponential'

def softmax_predictive_accuracy(logits_list, y):
    probs_list = [torch.softmax(logits, dim=1) for logits in logits_list]
    probs_tensor = torch.stack(probs_list, dim = 2)
    probs = torch.mean(probs_tensor, dim=2)
    _, pred_class = torch.max(probs, 1)
    correct = (pred_class == y)
    correct = correct.float().sum()
    return correct


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    total_correct = 0
    loss = None
    batch_size = args.batch_size
    epoch_size = len(train_loader.dataset)
    num_iters_in_epoch = len(train_loader)
    base_num_iter = (epoch - 1) * num_iters_in_epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if isinstance(optimizer, VOGN):
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                #pred = torch.softmax(output, dim=1)
                #correct = pred.eq(target.view_as(pred)).sum().item()
                return loss, output
        else:
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                #pred = torch.softmax(output)
                #correct = pred.eq(target.view_as(pred)).sum().item()
                return loss, output

        # update params
        _loss, _pred = optimizer.step(closure)
        loss = _loss.item()
        if isinstance(optimizer, VOGN):
            total_correct += softmax_predictive_accuracy(_pred, target)
        else:
            _pred = _pred.argmax(dim=1, keepdim=True)
            total_correct += _pred.eq(target.view_as(_pred)).sum().item()

        iteration = base_num_iter + batch_idx + 1

        if batch_idx % args.log_interval == 0:
            total_data_size = (batch_idx + 1) * batch_size
            accuracy = 100. * total_correct / total_data_size
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}/{} ({:.2f}%)'.format(
            #       epoch, total_data_size, epoch_size, 100. * (batch_idx + 1) / num_iters_in_epoch,
            #       loss, total_correct, total_data_size, accuracy))

            # write to log
            log = 'epoch,{},iteration,{},accuracy,{},loss,{}'.format(
               epoch, iteration, accuracy, loss
            )
            path = os.path.join(args.out, args.log_file_name)
            with open(path, 'a') as f:
                f.write(log + '\n')


        # if isinstance(optimizer, VOGN) and batch_idx % args.vogn_log_interval == 0:
        #     mu = optimizer.state['mu']
        #     mu_pre = getattr(optimizer, 'mu_pre')
        #     mu_upd_norm = mu.sub(mu_pre).norm().item()
        #     prec = optimizer.state['Precision']
        #     ggn = optimizer.state['GGN']
        #
        #     # write to log.vogn
        #     log = 'epoch,{},iteration,{},lr,{},' \
        #           'mu_mean,{},mu_std,{},mu_norm,{},' \
        #           'prec_mean,{},prec_std,{},prec_norm,{},' \
        #           'ggn_mean,{},ggn_std,{},ggn_norm,{},mu_upd_norm,{}'.format(
        #            epoch, iteration, lr,
        #            mu.mean(), mu.std(), mu.norm().item(),
        #            prec.mean(), prec.std(), prec.norm().item(),
        #            ggn.mean(), ggn.std(), ggn.norm().item(), mu_upd_norm)
        #     path = os.path.join(args.out, args.vogn_log_file_name)
        #     with open(path, 'a') as f:
        #         f.write(log + '\n')
        #
        #     if args.vogn_save_array:
        #         dirname = os.path.join(args.out, 'data')
        #         if not os.path.exists(dirname):
        #             os.makedirs(dirname)
        #         path = os.path.join(dirname, 'mu_iter{}.npy'.format(iteration))
        #         np.save(path, mu)
        #         path = os.path.join(dirname, 'prec_iter{}.npy'.format(iteration))
        #         np.save(path, prec)
        #         path = os.path.join(dirname, 'ggn_iter{}.npy'.format(iteration))
        #         np.save(path, ggn)

    accuracy = 100. * total_correct / epoch_size

    return accuracy, loss


def test(args, model, device, test_loader, optimzer=None):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)


            if optimzer is not None and isinstance(optimzer, VOGN):
                raw_noises = []
                for mc_sample in range(args.test_mc_samples):
                    raw_noises.append(torch.normal(mean=optimzer.state['mu'], std=1.0))
                outputs = optimzer.get_mc_predictions(model, data,
                                                      raw_noises=raw_noises)
                correct += softmax_predictive_accuracy(outputs, target)

            else:
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)



    return test_accuracy, test_loss


def main():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description='PyTorch Baseline Training')

    # Architecture Settings

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--depth', type=int, default=20, help='Model depth.')
    parser.add_argument('--block-name', type=str, default='BasicBlock',
                        help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
    parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')

    # Training Settings
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER_VOGN,
                        help='name of the optimizer')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default:128)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    # Hyper-parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='constant',
                        help='learning rate scheduler')
    parser.add_argument('--exponential_decay_rate', type=float, default=0.9,
                        help='multiplicative factor of learning rate decay')
    parser.add_argument('--beta1', type=float, default=0.999,
                        help='coefficient used for computing running average of gradient')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='coefficient used for computing running average of squared gradient')
    # VOGN Settings
    parser.add_argument('--prec_init', type=float, default=1.0,
                        help='initial precision for variational dist. q')
    parser.add_argument('--prior_prec', type=float, default=1.0,
                        help='prior precision on parameters')
    parser.add_argument('--mc_samples', type=int, default=1,
                        help='number of MC samples')
    parser.add_argument('--test_mc_samples', type=int, default=1,
                        help='number of MC samples for test')
    parser.add_argument('--normalize_prec', action='store_true', default=False,
                        help='for normalizing precision matrix')
    parser.add_argument('--normalize_mu', action='store_true', default=False,
                        help='for normalizing mu matrix')
    parser.add_argument('--mu_scale', type=float, default=10,
                        help='for normalizing mu matrix')
    parser.add_argument('--add_noise_for_test', action='store_true', default=False,
                        help='for adding noise to weights in test time')
    parser.add_argument('--vogn_log_interval', type=int, default=10,
                        help='how many batches to wait before logging VOGN status')
    parser.add_argument('--vogn_log_file_name', type=str, default='log.vogn',
                        help='log file name for VOGN parameters')
    parser.add_argument('--vogn_save_array', action='store_true', default=False,
                        help='for saving the arrays of VOGN')
    # Other
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='for saving the current model')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_file_name', type=str, default='log',
                        help='log file name')
    parser.add_argument('--param_log_file_name', type=str, default='log.data',
                        help='log file name for parameters')
    parser.add_argument('--out', type=str, default='result',
                        help='dir to save output files')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    args = parser.parse_args()

    # Setup device and random seed
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # Load Dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data = Dataset(args.dataset)
    train_loader = data.get_train_loader(batch_size=args.batch_size, shuffle=True)
    test_loader = data.get_test_loader(batch_size=args.test_batch_size, shuffle=True)
    num_classes = 10
    if args.dataset == 'cifar100':
        num_classes = 100
    # Build model
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            cardinality=args.cardinality,
            num_classes=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            growthRate=args.growthRate,
            compressionRate=args.compressionRate,
            dropRate=args.drop,
        )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            block_name=args.block_name,
        )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    model = model.to(device)

    # Setup optimizer
    if args.optimizer == OPTIMIZER_VOGN:
        optimizer = VOGN(model, train_set_size=len(train_loader.dataset),
                         lr=args.lr, beta1=args.beta1, beta2=args.beta2,
                         prec_init=args.prec_init, prior_prec=args.prior_prec,
                         num_samples=args.mc_samples)
    elif args.optimizer == OPTIMIZER_ADAM:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == OPTIMIZER_SGD:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError

    if args.lr_scheduler == LR_SCHEDULE_CONSTANT:
        scheduler = None
    elif args.lr_scheduler == LR_SCHEDULE_EXPONENTIAL:
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.exponential_decay_rate)
    else:
        raise ValueError

    print(args.dataset + ' Training')
    print('===========================')
    print('train data size: {}'.format(len(train_loader.dataset)))
    print('test data size: {}'.format(len(test_loader.dataset)))
    print('epochs: {}'.format(args.epochs))
    print('mini-batch size: {}'.format(args.batch_size))
    print('test mini-batch size: {}'.format(args.test_batch_size))
    print('arch: {}'.format(args.arch))
    print('optimizer: {}'.format(args.optimizer))
    print('learning rate: {}'.format(args.lr))
    print('learning rate scheduler: {}'.format(args.lr_scheduler))
    if args.lr_scheduler == LR_SCHEDULE_EXPONENTIAL:
        print('exponential decay rate: {}'.format(args.exponential_decay_rate))
    if args.optimizer in [OPTIMIZER_ADAM, OPTIMIZER_VOGN]:
        print('beta1: {}'.format(args.beta1))
        print('beta2: {}'.format(args.beta2))
    if args.optimizer == OPTIMIZER_VOGN:
        print('init precision: {}'.format(args.prec_init))
        print('prior precision: {}'.format(args.prior_prec))
        print('MC samples: {}'.format(args.mc_samples))
        print('test MC samples: {}'.format(args.test_mc_samples))
    if args.optimizer == OPTIMIZER_SGD:
        print('Momentum: {}'.format(args.momentum))
        print('Weight Decay: {}'.format(args.weight_decay))
    print('device: {}'.format(device))
    print('random seed: {}'.format(args.seed))
    print('===========================')

    # Copy this file to args.out
    shutil.copy(os.path.realpath(__file__), args.out)

    # Training
    for epoch in range(1, args.epochs + 1):

        # update learning rate
        if scheduler is not None:
            scheduler.step(epoch - 1)

        # train
        accuracy, loss = train(args, model, device, train_loader, optimizer, epoch)

        # test
        test_accuracy, test_loss = test(args, model, device, test_loader, optimizer)

        iteration = epoch * len(train_loader)
        log = 'epoch,{},iteration,{},' \
              'accuracy,{},loss,{},' \
              'test_accuracy,{},test_loss,{},' \
              'lr,{}'.format(
               epoch, iteration, accuracy, loss, test_accuracy, test_loss, optimizer.param_groups[0]['lr'])
        path = os.path.join(args.out, args.log_file_name)
        with open(path, 'a') as f:
            f.write(log + '\n')

    if args.save_model:
        path_model = os.path.join(args.out, '{}_{}.pt'.format(args.dataset, args.arch))
        path_optim = os.path.join(os.path.join(args.out, '{}_{}.opt'.format(args.dataset, args.arch)))
        torch.save(model.state_dict(), path_model)
        torch.save(optimizer.state, path_optim)


if __name__ == '__main__':
    main()
