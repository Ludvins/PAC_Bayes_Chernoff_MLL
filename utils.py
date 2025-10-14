import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import scipy
import random
from tqdm import tqdm
from laplace.utils import kron
import torch.distributions as dist
import copy
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
from torch.distributions.multivariate_normal import _precision_to_scale_tril


def non_latex_format():
    """ Set the matplotlib style to non-latex format """
    mpl.rcParams.update(mpl.rcParamsDefault)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.figsize"] = (16, 9)
    fontsize = 26
    matplotlib.rcParams.update({"font.size": fontsize})


def latex_format():
    """ Set the matplotlib style to latex format """
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{bm}",
        }
    )
    mpl.rc("font", family="Times New Roman")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.figsize"] = (16, 9)
    fontsize = 30
    matplotlib.rcParams.update({"font.size": fontsize})

def assert_reproducibility(seed: int):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def createmodel(k, random_seed, n_classes, n_channels):
    """ Create a LeNet5 model with k times the number of channels. 
    
    Arguments
    ---------
    k : int
        Multiplies the number of channels in the layers of LeNet-5.
    random_seed : int
                  Random number for reproducibility.
    n_classes : int
                Number of classes in the dataset.
    n_channels : int
                 Number of channels in the input data.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    return LeNet5(n_classes, n_channels, k)


class LeNet5(nn.Module):
    def __init__(self, n_classes, input_channels, k):
        """ Initialize the LeNet-5 model with k times the number of channels.
        
        The model has 3 convolutional layers, 2 pooling layers, and 2 fully connected layers.
        The first convolutional layer has int(6k) output channels.
        The second convolutional layer has int(16k) output channels.
        The third convolutional layer has int(120k) output channels.
        
        Arguments
        ---------
        n_classes : int
            Number of classes in the dataset.
        input_channels : int
            Number of channels in the input data.
        k : int
            Multiplicative factor of the number of channels.
        
        """
        super(LeNet5, self).__init__()

        self.part1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=int(6 * k),
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.part2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(6 * k),
                out_channels=int(16 * k),
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.part3 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(16 * k),
                out_channels=int(120 * k),
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(120 * k), out_features=int(84 * k)),
            nn.ReLU(),
            nn.Linear(in_features=int(84 * k), out_features=n_classes),
        )

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def MLPcreatemodel(random_seed, input_shape, hidden_sizes, n_classes):
    """ Create a MLP model 
    
    Arguments
    ---------
    input_shape : int
    hidden_sizes : array
		2d array with the number of neurons in each hidden layer.
    random_seed : int
                  Random number for reproducibility.
    n_classes : int
                Number of classes in the dataset.

    """

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    return MLP(input_shape, hidden_sizes, n_classes)



class MLP(nn.Module):
    def __init__(self, input_shape, hidden_sizes, n_classes):
        super(MLP, self).__init__()
        
        # Define layers
        layers = []
        # Flatten layer
        layers.append(nn.Flatten())
        
        # Input size after flattening
        input_size = torch.prod(torch.tensor(input_shape)).item()
        
        # Hidden layers with ReLU activations
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(previous_size, n_classes))
        
        # Combine all layers in the sequential container
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        """ Initialize the EarlyStopper object.
        
        Arguments
        ---------
        patience : int
            The number of iterations to wait before stopping training.
        min_delta : float
            The minimum delta between the current loss and the best loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """ Check if the training should stop.
        
        Arguments
        ---------
        validation_loss : float
            The loss on the validation set.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False



def train(model, name, train_loader, learning_rate, n_iters, device, criterion):
    """ Train the model using SGD with ExponentialLR scheduler and EarlyStopper
    
    Arguments
    ---------
    model : torch.nn.Module
        The model to train
    train_loader : torch.utils.data.DataLoader
        The data loader for the training set
    learning_rate : float
        The learning rate for the optimizer
    n_iters : int
        The number of iterations to train for
    device : torch.device
        The device to train on
    criterion : torch.nn.Module
        The loss function to optimize    
    """
    # Initialize EarlyStopper, optimizer and scheduler
    es = EarlyStopper(patience=4)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.95
    )
    # Initialize data iterator
    data_iter = iter(train_loader)
    iters_per_epoch = len(data_iter)
    aux_loss = 1

    # Lists to store metrics
    losses = []

    tq = tqdm(range(n_iters))
    for it in tq:
        # Set model to train mode
        model.train()

        # Get inputs and targets. If loader is exhausted, reinitialize.
        try:
            inputs, target = next(data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iter = iter(train_loader)
            inputs, target = next(data_iter)

        # Move data to device
        inputs = inputs.to(device)
        target = target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)  

        # Compute the loss
        loss = criterion(logits, target)  
        aux_loss += loss.detach().cpu().numpy()

        # Log the loss
        tq.set_postfix(
            {"Train cce": loss.detach().cpu().numpy(), "Patience": es.counter}
        )

        # Append loss to the list
        losses.append(loss.detach().cpu().numpy())

        # Backward pass
        loss.backward() 
        # Update the weights
        optimizer.step()

        # Step the scheduler and check for early stopping
        if it % iters_per_epoch == 0 and it != 0:
            scheduler.step()
            if aux_loss / iters_per_epoch < 0.01 or es.early_stop(aux_loss):
                break
            aux_loss = 0


    # Plot training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b', markersize=1)
    plt.title('Training Loss Curve '+ name)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    return model

def eval(device, model, loader, criterion):
    """ Evaluate the model on the loader using the criterion.
    
    Arguments
    ---------
    device : torch.device
        The device to evaluate on
    model : torch.nn.Module
        The model to evaluate
    loader : torch.utils.data.DataLoader
        The data loader to evaluate on
    criterion : torch.nn.Module
        The loss function to evaluate with
    """
    
    # Initialize counters
    correct = 0
    total = 0
    losses = 0
    
    # Set model to evaluation mode
    model.eval()
    
    # Iterate over the loader
    with torch.no_grad():
        for data, targets in loader:
            
            # Move data to device
            total += targets.size(0)
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(data)
            # Compute the probabilities
            probs = F.softmax(logits, dim=1)
            # Get the predicted class
            predicted = torch.argmax(probs, 1)
            # Update the counters
            correct += (predicted == targets).sum().detach().cpu().numpy()

            # Compute the loss
            loss = criterion(logits, targets) 
            # Update the loss
            losses += loss.detach().cpu().numpy() * targets.size(0)

    return 100 * correct / total, total, losses / total

def eval_laplace(device, laplace, loader, eps=1e-7):
    """ Evaluate the model on the loader using the criterion.
    
    Arguments
    ---------
    device : torch.device
        The device to evaluate on
    model : torch.nn.Module
        The model to evaluate
    loader : torch.utils.data.DataLoader
        The data loader to evaluate on
    """
    
    total = 0
    bayes_loss = 0
    gibbs_loss = 0
    correct_predictions = 0  # To track correct predictions for accuracy
    correct_gibbs_pred = 0
    generator = torch.Generator(device=device)

    # Iterate over the loader
    with torch.no_grad():
        for data, targets in loader:

            # Move data to device
            total += targets.size(0)
            data = data.to(device)
            targets = targets.to(device)

            # To avoid softmax computation, set likelihood to regression
            laplace.likelihood = "regression"

            # (n_samples, batch_size, output_shape) - Samples are logits
            logits_samples = laplace.predictive_samples(data, pred_type="glm", n_samples=512)
            
            # Get probabilities of true classes
            oh_targets = F.one_hot(targets, num_classes=logits_samples.size(-1))  
    
            log_prob = torch.sum(logits_samples * oh_targets, -1) \
                - torch.logsumexp(logits_samples, -1)

            # Compute Bayesian and Gibbs loss
            bayes_loss -= torch.logsumexp(log_prob - torch.log(torch.tensor(512, device=device)), 0).sum()
            gibbs_loss -= log_prob.mean(0).sum()

            avg_logits = logits_samples.mean(0)  # Mean over samples (dimension 0)
            bma_predictions = avg_logits.argmax(dim=-1)  # Get predicted class (argmax over logits)

            predictions_per_model = torch.argmax(logits_samples, dim=-1)
            gibbs_predictions, _ = torch.mode(predictions_per_model, dim=0)
            assert gibbs_predictions.shape == targets.shape, "Shapes of predictions and targets must match!"
            correct_gibbs_pred += (gibbs_predictions == targets).sum().item()

            # Compare predictions to targets and accumulate correct predictions
            correct_predictions += (bma_predictions == targets).sum().item()

    # Compute accuracy
    bma_accuracy = correct_predictions / total
    gibbs_accuracy = correct_gibbs_pred / total
    return bayes_loss / total, gibbs_loss / total, bma_accuracy, gibbs_accuracy



def eval_extended_laplace(device, laplace, loader, post_variance=0.001, eps=1e-7):
    """ Evaluate the model on the loader using the criterion.

    Arguments
    ---------
    device : torch.device
        The device to evaluate on
    model : torch.nn.Module
        The model to evaluate
    loader : torch.utils.data.DataLoader
        The data loader to evaluate on
    """

    total = 0
    bayes_loss = 0
    gibbs_loss = 0
    correct_predictions = 0  # To track correct predictions for accuracy
    correct_gibbs_pred = 0
    generator = torch.Generator(device=device)

    # Iterate over the loader
    with torch.no_grad():
        for data, targets in loader:

            # Move data to device
            total += targets.size(0)
            data = data.to(device)
            targets = targets.to(device)

            # To avoid softmax computation, set likelihood to regression
            laplace.likelihood = "regression"

            # Store non-laplace original params
            original_params = [p.clone() for p in list(reversed(list(laplace.model.parameters())))[2:]]
            
            # Adds noise to the non-laplace parameters
            for params in list(reversed(list(laplace.model.parameters())))[2:]:
                noise = torch.randn_like(params)*post_variance
                params.add_(noise)

            # (n_samples, batch_size, output_shape) - Samples are logits
            logits_samples = laplace.predictive_samples(data, pred_type="glm", n_samples=512)
            
            # Restore the original parameters
            for p, orig in zip(list(reversed(list(laplace.model.parameters())))[2:], original_params):
                p.copy_(orig)

            # Get probabilities of true classes
            oh_targets = F.one_hot(targets, num_classes=logits_samples.size(-1))

            log_prob = torch.sum(logits_samples * oh_targets, -1) \
                - torch.logsumexp(logits_samples, -1)

            # Compute Bayesian and Gibbs loss
            bayes_loss -= torch.logsumexp(log_prob - torch.log(torch.tensor(512, device=device)), 0).sum()
            gibbs_loss -= log_prob.mean(0).sum()

            avg_logits = logits_samples.mean(0)  # Mean over samples (dimension 0)
            bma_predictions = avg_logits.argmax(dim=-1)  # Get predicted class (argmax over logits)
            
            predictions_per_model = torch.argmax(logits_samples, dim=-1)
            gibbs_predictions, _ = torch.mode(predictions_per_model, dim=0)
            assert gibbs_predictions.shape == targets.shape, "Shapes of predictions and targets must match!"
            correct_gibbs_pred += (gibbs_predictions == targets).sum().item()

            # Compare predictions to targets and accumulate correct predictions
            correct_predictions += (bma_predictions == targets).sum().item()

    # Compute accuracy
    bma_accuracy = correct_predictions / total
    gibbs_accuracy = correct_gibbs_pred / total
    return bayes_loss / total, gibbs_loss / total, bma_accuracy, gibbs_accuracy



def compute_expected_norm(laplace, num_samples=512):
    """
    Compute the expected L2 norm of the last layer's parameters with samples from Laplace posterior.
    Args:
        laplace: Laplace object containing the posterior distribution.
        num_samples: Number of posterior samples.
    Returns:
        Expected L2 norm.
    """
    samples = laplace.sample(n_samples=num_samples)
    norms = torch.norm(samples, p=2, dim=1)

    return norms.mean()


def get_sampled_model(laplace):
    """
    Returns a new model with sampled last-layer weights from the Laplace posterior.
    Works only for last-layer Laplace approximations.
    """
    sampled_params = laplace.sample(n_samples=1)[0]
    sampled_bias = sampled_params[-10:]
    sampled_weights = sampled_params[:-10]
    
    sampled_model = copy.deepcopy(laplace.model)
    
    for module in reversed(list(sampled_model.modules())):
        module.bias.data = sampled_bias
        module.weight.data = sampled_weights.reshape(module.weight.data.shape)
        break 

    return sampled_model 



def compute_expected_input_gradient_norm(laplace, data_loader, n_models=20, device='cuda'):

    # Only works for linear last-layers

    laplace.model.eval()
    total_norm = 0.0
    num_samples = 0

    for inputs, targets in data_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs.requires_grad = True

        batch_norms = []
        
        for _ in range(n_models):

            model = get_sampled_model(laplace)
            
            outputs = model(inputs)
            ce = nn.CrossEntropyLoss()
            loss = ce(outputs, targets)
            grads = torch.autograd.grad(loss, inputs, grad_outputs=torch.ones_like(loss), create_graph=False)[0]

            norm_batch = torch.norm(grads.view(grads.shape[0], -1), p=2, dim=1)**2 #Shape: (batch_size,)
            batch_norms.append(norm_batch)

        batch_norms = torch.stack(batch_norms, dim=0)

        total_norm += batch_norms.sum().item()
        num_samples += inputs.shape[0]

    return total_norm/(num_samples*n_models)

def kl_compare_blocks_vs_full(laplace):
    """
    Reconstruct each block's precision using the code snippet, compute
    the KL contribution manually, then compare with the full NxN matrix approach.
    """

    # 1) Basic definitions
    sigma = laplace.prior_precision       # Scalar prior precision
    n = laplace.mean.numel()              # Total parameter dimension
    mean_full = laplace.mean              # Posterior mean

    # We'll accumulate three sums for the block-based approach
    sum_trace_cov = 0.0
    sum_mu_norm =   0.0
    sum_logdet    = 0.0

    offset = 0  # To slice the correct part of the mean for each block

    # "exponent" is 1 
    exponent = 1

    # 2) Build blocks (precision matrices) using the snippet
    for Qs, ls, delta in zip(laplace.posterior_precision.eigenvectors,
                             laplace.posterior_precision.eigenvalues,
                             laplace.posterior_precision.deltas):

        if len(ls) == 1:
            # Single-eigenvalue block
            Q, eigval = Qs[0], ls[0]
            # block_prec will be shape [d, d]
            block_prec = Q @ torch.diag(torch.pow(eigval + delta, exponent)) @ Q.T
        else:
            # 2D Kronecker block
            Q1, Q2 = Qs
            l1, l2 = ls
            Q = torch.kron(Q1, Q2)
            if laplace.posterior_precision.damping:
                delta_sqrt = torch.sqrt(delta)
                eigval = torch.pow(
                    torch.outer(l1 + delta_sqrt, l2 + delta_sqrt), exponent
                )
            else:
                eigval = torch.pow(torch.outer(l1, l2) + delta, exponent)
            L = torch.diag(eigval.flatten())
            block_prec = Q @ L @ Q.T

        
        d = block_prec.shape[0]
        # Get the eigenvalues of block_prec for fast calculation of the trace
        if len(ls) == 1:
            block_prec_eigvals = (ls[0] + delta)**exponent
        else:
            block_prec_eigvals = torch.pow(torch.outer(l1, l2) + delta, exponent).flatten()

        # 3a) trace(Sigma_i^-1) in the KL formula actually means trace(block_prec) if block_cov is Sigma_i.
        #     But the formula requires "sigma * sum_i trace(Sigma_i^-1)" => "sigma * trace(block_cov^-1)?"
        #     Careful: "Sigma_i^-1" is block_prec, so "trace(Sigma_i^-1)" = trace(block_prec).
        # We will handle the multiplication by "sigma" later in the final expression.

        trace_block_cov = (1./block_prec_eigvals).sum()

        # 3b) mu_i^T  mu_i
        mu_block = laplace.mean[offset : offset + d]  # shape [d]
        offset += d

        mu_norm = torch.norm(mu_block, p=2)**2

        # 3c) ln det(Sigma_i) => ln det(block_cov)
        #     Since block_prec = Sigma_i^-1,  det(Sigma_i) = 1 / det(block_prec).
        # => ln det(Sigma_i) = - ln det(block_prec).
        # We'll compute logdet(block_prec) and then put a minus sign
        logdet_block_prec = torch.logdet(block_prec)

        # 3d) Accumulate in the big sums
        sum_trace_cov += trace_block_cov
        sum_mu_norm += mu_norm
        sum_logdet += logdet_block_prec

    if torch.isnan(sum_logdet).any():
        print("Logdet contains NaNs")

    # 4) Combine for the final block-based KL
    # The standard formula is:
    #
    #  KL = 0.5 * [
    #     sigma * sum_i( trace(Sigma_i^-1) )
    #   + sum_i( mu_i^T Sigma_i^-1 mu_i )
    #   + sum_i( ln det(Sigma_i) )
    #   - n ln sigma
    #   - n
    # ]
    #
    # Where sum_i trace(Sigma_i^-1) = sum_trace_inv
    # and sum_i ln det(Sigma_i) = sum_logdet, etc.

    kl_blocks = 0.5 * (
          1./sigma * sum_trace_cov
        + 1./sigma * sum_mu_norm
        + sum_logdet
        + n*torch.log(sigma)
        - n
    )

    # 5) Compare with the full NxN approach
    #    We'll reconstruct the full precision matrix with .to_matrix(),
    #    invert it, and compute the KL in one shot.
    #prec_full = laplace.posterior_precision.to_matrix()  # shape [n,n]
    #sigma_full = torch.inverse(prec_full)    

    # 5a) sigma * trace(Sigma_full)
    #trace_term = 1./sigma * torch.trace(sigma_full)

    # 5b) mu^T mu
    #full_mu_norm = 1./sigma * torch.norm(mean_full, p=2)**2

    # 5c) ln det(Sigma_full)
    #logdet_Sigma_full = torch.logdet(prec_full).item()

    #kl_full = 0.5 * (
    #      trace_term
    #    + full_mu_norm
    #    + logdet_Sigma_full
    #    + n*torch.log(sigma)
    #    - n
    #)

    # 6) Print or return for comparison
    print(f"KL (block-by-block) = {kl_blocks.item():.6f}")
    #print(f"KL (full matrix)    = {kl_full.item():.6f}")

    return kl_blocks


def estimate_kl(laplace, num_samples=1024):

    # Extract parameters
    scalar_prec_prior = laplace.prior_precision 
    
    mean_posterior = laplace.mean
    prec_posterior = laplace.posterior_precision.to_matrix()
    L = _precision_to_scale_tril(prec_posterior)

    #prec_posterior = make_psd(prec_posterior)

    mean_prior = torch.zeros(mean_posterior.shape[0], device=mean_posterior.device)
    prec_prior = scalar_prec_prior*torch.eye(mean_posterior.shape[0], device=mean_posterior.device)
    
    # Define the distributions
    prior_dist = dist.MultivariateNormal(mean_prior, precision_matrix=prec_prior)
    posterior_dist = dist.MultivariateNormal(mean_posterior, scale_tril=L)

    # Sample from the posterior
    samples_posterior = posterior_dist.sample((num_samples,))

    # Compute the log probabilities under the prior and posterior
    log_prob_posterior = posterior_dist.log_prob(samples_posterior)
    log_prob_prior = prior_dist.log_prob(samples_posterior)

    # Estimate KL divergence
    kl_divergence = (log_prob_posterior - log_prob_prior).mean()

    return kl_divergence.item()


def extended_kl(laplace, last_layer_params, posterior_precision=1000):
    
    params = torch.cat([p.detach().flatten() for p in laplace.model.parameters()])
    
    prior_prec = laplace.prior_precision
    posterior_mean = params[:-last_layer_params]
    d = len(list(posterior_mean))
    posterior_mean_norm = torch.norm(posterior_mean, p=2)

    return 0.5*(d*posterior_precision/prior_prec + posterior_precision*posterior_mean_norm**2 - d +d*torch.log(prior_prec/posterior_precision))
    



def get_log_p(device, model, loader):
    cce = nn.CrossEntropyLoss(reduction="none")  # supervised classification loss
    aux = []
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            logits = model(data)
            log_p = -cce(logits, targets)  # supervised loss
            aux.append(log_p)
    return torch.cat(aux)


def get_log_p(device, laplace, loader, post_variance=0.001, eps=1e-7):
    """ Evaluate the model on the loader using the criterion.
    
    Arguments
    ---------
    device : torch.device
        The device to evaluate on
    model : torch.nn.Module
        The model to evaluate
    loader : torch.utils.data.DataLoader
        The data loader to evaluate on
    """
    
    # Initialize counters
    total = 0
    log_p = []
    
    # Iterate over the loader
    with torch.no_grad():
        for data, targets in loader:
            
            # Move data to device
            total += targets.size(0)
            data = data.to(device)
            targets = targets.to(device)
            
            # To avoid softmax computation
            laplace.likelihood = "regression"

            # (n_samples, batch_size, output_shape) - Samples are logits
            logits_samples = laplace.predictive_samples(data, pred_type="glm", n_samples=512)

            # Get probabilities of true classes
            oh_targets = F.one_hot(targets, num_classes=10)
            
            log_prob = torch.sum(logits_samples * oh_targets, -1) \
                - torch.logsumexp(logits_samples, -1)
            
            log_p.append(log_prob)

    return torch.cat(log_p, 1)


def rate_function_inv(log_p, s_value, device):
  min_lamb=torch.tensor(0).to(device)
  max_lamb=torch.tensor(300000).to(device)

  s_value=torch.tensor(s_value).to(device)
  inv, lamb, J = aux_inv_rate_function_TernarySearch(log_p, s_value, min_lamb, max_lamb, 0.5, device)
  
  return inv,  lamb, J


def aux_inv_rate_function_TernarySearch(log_p, s_value, low, high, epsilon, device):

    while (high - low) > epsilon:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3
        if eval_inverse_rate_at_lambda(log_p, mid1, s_value, device) < eval_inverse_rate_at_lambda(log_p, mid2, s_value, device):
            high = mid2
        else:
            low = mid1
    # Return the midpoint of the final range
    mid = (low + high) / 2
    inv = eval_inverse_rate_at_lambda(log_p, mid, s_value, device)
    
    return [
        inv.detach().cpu().numpy(),
        mid.detach().cpu().numpy(),
        (inv*mid - s_value).detach().cpu().numpy(),
    ]

def eval_inverse_rate_at_lambda(log_p, lamb, s_value, device):
    jensen_val = (
        torch.logsumexp(lamb * log_p, -1) - torch.log(torch.tensor(log_p.shape[-1], device=device)) - lamb*torch.mean(log_p, -1)
    ).mean(0)
    print(jensen_val)
    # aux_tensor = torch.log(torch.tensor(10/0.05, device=device))
    # jensen_val = torch.log(torch.exp(jensen_val) + torch.sqrt(0.5*aux_tensor/torch.tensor(log_p.shape[-1], device=device))).mean(0)
    return (s_value + jensen_val)/lamb

def compute_trace(krondecomposed):
    trace_term = 0
    n_params = 0
    for Qs, ls, delta in zip(krondecomposed.eigenvectors, krondecomposed.eigenvalues, krondecomposed.deltas):
        if len(ls) == 1:
            Q, l = Qs[0], ls[0]
            block = Q @ torch.diag(torch.pow(l + delta, -1)) @ Q.T
        else:
            Q1, Q2 = Qs
            l1, l2 = ls
            Q = kron(Q1, Q2)
            if krondecomposed.damping:
                delta_sqrt = torch.sqrt(delta)
                l = torch.pow(
                    torch.outer(l1 + delta_sqrt, l2 + delta_sqrt), -1
                )
            else:
                l = torch.pow(torch.outer(l1, l2) + delta, -1)
            L = torch.diag(l.flatten())
            block = Q @ L @ Q.T
        
        block = torch.linalg.inv(block + 1e-3 * torch.eye(block.shape[0]).to(block.device))

        trace_term += torch.trace(block)
        n_params += block.shape[0]
        
    return trace_term, n_params

# def aux_rate_function_TernarySearch(log_p, s_value, low, high, epsilon):

#     while (high - low) > epsilon:
#         mid1 = low + (high - low) / 3
#         mid2 = high - (high - low) / 3

#         if eval_log_p(log_p, mid1, s_value) < eval_log_p(log_p, mid2, s_value):
#             low = mid1
#         else:
#             high = mid2

#     # Return the midpoint of the final range
#     mid = (low + high) / 2
#     return [
#         eval_log_p(log_p, mid, s_value).detach().cpu().numpy(),
#         mid.detach().cpu().numpy(),
#         (mid * s_value - eval_log_p(log_p, mid, s_value)).detach().cpu().numpy(),
#     ]


def eval_cummulant(log_p, lambdas, device):
    # log_p shape (samples, n_data)
    
    return np.array(
        [
            (
                torch.logsumexp(lamb * log_p, -1)
                - torch.log(torch.tensor(log_p.shape[-1], device=device))
                - torch.mean(lamb * log_p, -1)
            )
            .detach()
            .cpu()
            .numpy().mean(0)
            for lamb in lambdas
        ]
    )


def inverse_rate_function(model, lambdas, rate_vals):
    jensen_vals = eval_cummulant(model, lambdas)

    return np.array([np.min((jensen_vals + rate) / lambdas) for rate in rate_vals])



def get_log_p_mfvi(device, mfvi_model, loader, num_samples=128):
    """Get log probabilities from MFVI model samples"""
    log_probs = []
    
    # Iterate over the loader
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            
            # Get logits samples from MFVI model
            # Shape: (num_samples, batch_size, num_classes)
            logits_samples = mfvi_model.predict(data)
            
            # Get probabilities of true classes using one-hot encoding
            oh_targets = F.one_hot(targets, num_classes=logits_samples.size(-1))
            
            # Compute log probabilities
            log_prob = torch.sum(logits_samples * oh_targets, -1) \
                      - torch.logsumexp(logits_samples, -1)
            
            log_probs.append(log_prob)
    
    # Concatenate all batches
    # Shape: (num_samples, total_data_points)
    return torch.cat(log_probs, 1)

def rate_function_inv_mfvi(mfvi_model, loader, s_value, device, num_samples=128):
    """Compute rate function inverse for MFVI model"""
    # Get log probabilities using MFVI sampling
    log_p = get_log_p_mfvi(device, mfvi_model, loader, num_samples)
    
    # Use the existing rate_function_inv with the MFVI samples
    return rate_function_inv(log_p, s_value, device)