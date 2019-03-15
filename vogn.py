import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn as nn
import torch.nn.functional as F

################################
## PyTorch Optimizer for VOGN ##
################################
required = object()


def update_input(self, input, output):
    self.input = input[0].data
    self.output = output

class VOGN(Optimizer):
    """Implements the VOGN algorithm. It uses the Generalized Gauss Newton (GGN)
        approximation to the Hessian and a mean-field approximation. Note that this
        optimizer does **not** support multiple model parameter groups. All model
        parameters must use the same optimizer parameters.
        model (nn.Module): network model
        train_set_size (int): number of data points in the full training set
            (objective assumed to be on the form (1/M)*sum(-log p))
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): coefficient used for computing
            running average of squared gradient (default: 0.999)
        prior_prec (float, optional): prior precision on parameters
            (default: 1.0)
        prec_init (float, optional): initial precision for variational dist. q
            (default: 1.0)
        num_samples (float, optional): number of MC samples
            (default: 1)
    """

    def __init__(self, model, train_set_size, lr=1e-3, beta1=0.9, beta2=0.999, prior_prec=1.0, prec_init=1.0,
                 num_samples=1):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if prior_prec < 0.0:
            raise ValueError("Invalid prior precision value: {}".format(prior_prec))
        if prec_init < 0.0:
            raise ValueError("Invalid initial s value: {}".format(prec_init))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta2))
        if num_samples < 1:
            raise ValueError("Invalid num_samples parameter: {}".format(num_samples))
        if train_set_size < 1:
            raise ValueError("Invalid number of training data points: {}".format(train_set_size))

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, prior_prec=prior_prec, prec_init=prec_init,
                        num_samples=num_samples,
                        train_set_size=train_set_size)
        self.train_modules = []
        self.set_train_modules(model)
        super(type(self), self).__init__(model.parameters(), defaults)
        for module in self.train_modules:
            module.register_forward_hook(update_input)

        p = parameters_to_vector(self.param_groups[0]['params'])
        # mean parameter of variational distribution.
        self.state['mu'] = p.clone().detach()
        self.state['mu_grad_avg'] = torch.zeros_like(p).detach()
        # covariance parameter of variational distribution -- saved as the diagonal precision matrix.
        self.state['Precision'] = torch.ones_like(p).mul_(defaults['prec_init']).detach()

    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss without doing the backward pass
        """

        if closure is None:
            raise RuntimeError(
                'For now, VOGN only supports that the model/loss can be reevaluated inside the step function')

        defaults = self.defaults
        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']

        Precision = self.state['Precision']
        mu = self.state['mu']
        GGN_hat = None
        mu_grad_hat = None
        loss_list = []
        pred_list = []
        for _ in range(defaults['num_samples']):
            # Sample a parameter vector:
            raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0).detach()
            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(Precision))
            vector_to_parameters(p, parameters)

            # Store the loss
            loss, preds = closure()
            loss_list.append(loss.detach())
            pred_list.append(preds.detach())
            linear_combinations = []
            # Store the pre-activations
            for module in self.train_modules:
                linear_combinations.append(module.output)

            # Do the Backward pass for gradients and square gradients
            linear_grad = torch.autograd.grad(loss, linear_combinations)
            ggn = []
            grad = []
            N = 1
            for i, module in enumerate(self.train_modules):
                G = linear_grad[i].detach()
                A = module.input.detach()
                M = A.shape[0]
                A = torch.cat([A] * N)
                G *= M
                G2 = torch.mul(G, G)

                if isinstance(module, nn.BatchNorm1d):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ij->j', torch.mul(G, A)))
                    ggn.append(torch.einsum('ij->j', torch.mul(G2, A2)))
                    if module.bias is not None:
                        grad.append(torch.einsum('ij->j', G))
                        ggn.append(torch.einsum('ij->j', G2))

                if isinstance(module, nn.BatchNorm2d):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ijkl->j', torch.mul(G, A)))
                    ggn.append(torch.einsum('ijkl->j', torch.mul(G2, A2)))
                    if module.bias is not None:
                        grad.append(torch.einsum('ijkl->j', G))
                        ggn.append(torch.einsum('ijkl->j', G2))

                if isinstance(module, nn.Linear):
                    A2 = torch.mul(A, A)
                    grad.append(torch.einsum('ij,ik->jk', G, A))
                    ggn.append(torch.einsum('ij, ik->jk', G2, A2))
                    if module.bias is not None:
                        # A = torch.ones((M * N, 1), device=A.device)
                        grad.append(torch.einsum('ij->j', G))
                        ggn.append(torch.einsum('ij->j', G2))

                if isinstance(module, nn.Conv2d):
                    A = F.unfold(A, kernel_size=module.kernel_size, dilation=module.dilation, padding=module.padding,
                                 stride=module.stride)
                    A2 = torch.mul(A, A)
                    _, k, hw = A.shape
                    _, c, _, _ = G.shape
                    '''G = G.view(M, c, -1)
                    mean = torch.zeros((c, k), device=A.device)
                    mean.addbmm_(G, A)'''
                    _, c, _, _ = G.shape
                    G = G.view(M * N, c, -1)
                    G2 = G2.view(M * N, c, -1)
                    grad.append(torch.einsum('ijl,ikl->jk', G, A))
                    '''mean = torch.zeros((c, k), device=A.device)
                    mean.addbmm_(torch.mul(G, G), torch.mul(A, A))'''
                    ggn.append(torch.einsum('ijl,ikl->jk', G2, A2))
                    if module.bias is not None:
                        # A = torch.ones((M * N, 1, hw), device=A.device)
                        '''mean = torch.zeros((c, 1), device=A.device)
                        mean.addbmm_(G, A)'''
                        grad.append(torch.einsum('ijl->j', G).detach())
                        '''mean = torch.zeros((c, 1), device=A.device)
                        mean.addbmm_(torch.mul(G, G), torch.mul(A, A))'''
                        ggn.append(torch.einsum('ijl->j', G2).detach())
            grad_vec = parameters_to_vector(grad).div(M).detach()
            ggn_vec = parameters_to_vector(ggn).div(M).detach()

            if mu_grad_hat is None:
                mu_grad_hat = torch.zeros_like(grad_vec)
            mu_grad_hat.add_(grad_vec)

            if GGN_hat is None:
                GGN_hat = torch.zeros_like(ggn_vec)
            GGN_hat.add_(ggn_vec)



            # Convert the parameter gradient to a single vector.
        mu_grad_hat = mu_grad_hat.mul(defaults['train_set_size'] / defaults['num_samples'])
        GGN_hat.mul_(defaults['train_set_size'] / defaults['num_samples'])
        self.state['mu_grad_avg'].mul_(defaults['beta1']).add_(mu_grad_hat.mul(1 - defaults['beta1']))
        # Get the mean loss over the number of samples
        loss = torch.mean(torch.stack(loss_list))
        #preds = torch.mean(torch.stack(pred_list))
        # Update precision matrix
        Precision = Precision.mul(defaults['beta2']) + GGN_hat.add(defaults['prior_prec']).mul(1 - defaults['beta2'])
        self.state['Precision'] = Precision
        # Update mean vector
        mu.addcdiv_(-self.param_groups[0]['lr'], self.state['mu_grad_avg'] + torch.mul(mu, defaults['prior_prec']), Precision)
        self.state['mu'] = mu
        vector_to_parameters(self.state['mu'], self.param_groups[0]['params'])
        return loss, pred_list

    def get_mc_predictions(self, forward_function, inputs, ret_numpy=False, raw_noises=None, *args, **kwargs):
        """Returns Monte Carlo predictions.
        Arguments:
            forward_function (callable): The forward function of the model
                that takes inputs and returns the outputs.
            inputs (FloatTensor): The inputs to the model.
            mc_samples (int): The number of Monte Carlo samples.
            ret_numpy (bool): If true, the returned list contains numpy arrays,
                otherwise it contains torch tensors.
        """

        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        predictions = []
        Precision = self.state['Precision']
        mu = self.state['mu']
        if raw_noises is None:
            raw_noises = [torch.zeros_like(mu)]
        for raw_noise in raw_noises:
            # Sample a parameter vector:
            # raw_noise = torch.normal(mean=torch.zeros_like(mu), std=1.0)

            p = torch.addcdiv(mu, 1., raw_noise, torch.sqrt(Precision))
            vector_to_parameters(p, parameters)

            # Call the forward computation function
            outputs = forward_function(inputs, *args, **kwargs)
            if ret_numpy:
                outputs = outputs.data.cpu().numpy()
            predictions.append(outputs)
        vector_to_parameters(self.state['mu'], self.param_groups[0]['params'])
        return predictions

    def _kl_gaussian(self, p_mu, p_sigma, q_mu, q_sigma):
        var_ratio = (p_sigma / q_sigma).pow(2)
        t1 = ((p_mu - q_mu) / q_sigma).pow(2)
        return 0.5 * torch.sum((var_ratio + t1 - 1 - var_ratio.log()))

    def kl_divergence(self):
        prec0 = self.defaults['prior_prec']
        prec = self.state['Precision']
        mu = self.state['mu']
        sigma = 1. / torch.sqrt(prec)
        mu0 = 0.
        sigma0 = 1. / math.sqrt(prec0)
        kl = self._kl_gaussian(p_mu=mu, p_sigma=sigma, q_mu=mu0, q_sigma=sigma0)
        return kl