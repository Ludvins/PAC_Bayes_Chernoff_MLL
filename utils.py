import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from laplace.utils import kron


import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl


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
    es = EarlyStopper(patience=2)
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
    
    # Initialize counters
    total = 0
    bayes_loss = 0
    gibbs_loss = 0
    
    # Iterate over the loader
    with torch.no_grad():
        for data, targets in loader:
            
            # Move data to device
            total += targets.size(0)
            data = data.to(device)
            targets = targets.to(device)
            
            # To avoid softmax computation
            laplace.likelihood = "regression"
            
            # (n_samples, batch_size, output_shape) Samples are logits
            logits_samples = laplace.predictive_samples(data, pred_type = "glm", n_samples = 512)
            # Get probabilities of true classes
            oh_targets = F.one_hot(targets, num_classes=10)
            
            log_prob = torch.sum(logits_samples * oh_targets, -1) \
                - torch.logsumexp(logits_samples, -1)
            
            
            bayes_loss -= torch.logsumexp(log_prob - torch.log(torch.tensor(512, device=device)), 0).sum()
            gibbs_loss -= log_prob.mean(0).sum()
            

    return bayes_loss/total, gibbs_loss/total

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


def get_log_p(device, laplace, loader, eps=1e-7):
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
            
            # (n_samples, batch_size, output_shape) Samples are logits
            logits_samples = laplace.predictive_samples(data, pred_type = "glm", n_samples = 512)
            # Get probabilities of true classes
            oh_targets = F.one_hot(targets, num_classes=10)
            
            log_prob = torch.sum(logits_samples * oh_targets, -1) \
                - torch.logsumexp(logits_samples, -1)
            
            log_p.append(log_prob)

    return torch.cat(log_p, 1)
# #Binary Search for lambdas
# def rate_function(log_p, s_value, device):
#   if (s_value<0):
#     min_lamb=torch.tensor(-10000).to(device)
#     max_lamb=torch.tensor(0).to(device)
#   else:
#     min_lamb=torch.tensor(0).to(device)
#     max_lamb=torch.tensor(10000).to(device)

#   s_value=torch.tensor(s_value).to(device)
#   return aux_rate_function_TernarySearch(log_p, s_value, min_lamb, max_lamb, 0.001)

#Binary Search for lambdas
def rate_function_inv(log_p, s_value, device):
  min_lamb=torch.tensor(0).to(device)
  max_lamb=torch.tensor(100000).to(device)

  s_value=torch.tensor(s_value).to(device)
  inv, _, _ = aux_inv_rate_function_TernarySearch(log_p, s_value, min_lamb, max_lamb, 0.01, device)
  
  return inv


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
    print(mid, inv)
    return [
        inv.detach().cpu().numpy(),
        mid.detach().cpu().numpy(),
        (inv*mid - s_value).detach().cpu().numpy(),
    ]

def eval_inverse_rate_at_lambda(log_p, lamb, s_value, device):
    jensen_val = (
        torch.logsumexp(lamb * log_p, -1)
        - torch.log(torch.tensor(log_p.shape[-1], device=device))
    ).mean(0)

    # aux_tensor = torch.log(torch.tensor(10/0.05, device=device))
    # jensen_val = torch.log(torch.exp(jensen_val) + torch.sqrt(0.5*aux_tensor/torch.tensor(log_p.shape[-1], device=device))).mean(0)
    return (s_value + jensen_val)/lamb - torch.mean(log_p)

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



