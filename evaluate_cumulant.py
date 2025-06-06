import torch.nn as nn
import torch
from torchvision import datasets, transforms
import numpy as np
import torchvision
import pickle
import matplotlib.pyplot as plt
from laplace import Laplace
from utils import latex_format, assert_reproducibility,  get_log_p, rate_function_inv, compute_expected_norm, compute_expected_input_gradient_norm, extended_kl, eval_extended_laplace
from torch.nn.utils import vector_to_parameters
import pandas as pd
from tqdm import tqdm
# Activate Latex format for matplotlib
latex_format()

# Hyper-Parameters
RANDOM_SEED = 15
LEARNING_RATE = 0.01
SUBSET_SIZE = 50000
TEST_SUBSET_SIZE = 10000
N_ITERS = 2000000
BATCH_SIZE = 200
BATCH_SIZE_TEST = 1000
IMG_SIZE = 32
N_CLASSES = 10
WEIGHT_DECAY = 0.01

assert_reproducibility(RANDOM_SEED)

import argparse
parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20
parser.add_argument("--hessian", help="csv file path", type=str)
parser.add_argument("--subset", help="csv file path", type=str)
parser.add_argument("--p", help="csv file path", type=float)

args = parser.parse_args()

if args.p == 0:
    prior = "opt"
else:
    prior = "fix"

# setup devices
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(RANDOM_SEED)
else:
    device = torch.device("cpu")

#######################################################################
############################# DATASET #################################
#######################################################################

transforms = torchvision.transforms.Compose([
                  torchvision.transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


train_dataset = datasets.CIFAR10(root='cifar_data',
                                train=True,
                                transform=transforms,
                                download=True)

test_dataset = datasets.CIFAR10(root='cifar_data',
                                train=False,
                                transform=transforms)

train_dataset = torch.utils.data.Subset(train_dataset, list(range(0, SUBSET_SIZE)))
test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, TEST_SUBSET_SIZE)))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE_TEST,
                          shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE_TEST,
                          shuffle=False)

#######################################################################
############################# TRAIN MODELS ############################
#######################################################################

model_type='ConvNN'


labels = np.loadtxt(f"models/{model_type}_model_labels.txt", delimiter=" ", dtype = str)

n_params = np.loadtxt(f"models/{model_type}_n_params.txt")



subset = "last_layer"
hessian = "kron"
delta = 0.05

csv_path = f"results/laplace_{model_type}_{subset}_{hessian}_scalar_{args.p}_results.csv"
results_laplace = pd.read_csv(csv_path)
inverse_rates = []
s_values = []
variances = []
lambdas = []
expected_cumulant = []
expected_norms = []
expected_input_grads = []
expanded_kl = []


g_cpu = torch.Generator(device=device)
g_cpu.manual_seed(RANDOM_SEED)

with tqdm(range(len(n_params))) as t:
  for i in range(len(labels)):
    t.set_description(f"Model {labels[i]}")

    with open(f"models/{labels[i]}.pickle", "rb") as handle:
      model = pickle.load(handle)
      la = Laplace(model, "classification",
                        subset_of_weights=subset,
                        hessian_structure=hessian)
      la.load_state_dict(torch.load(f'laplace_models/{labels[i]}_{subset}_{hessian}_scalar_state_dict.pt'))
      if prior != "opt":
        la.prior_precision = args.p

      log_p = get_log_p(device, la, test_loader, eps=1e-7)

      last_layer_params = results_laplace.query(f"model=='{labels[i]}'")["last layer params"].item()
      expand_kl = extended_kl(la, last_layer_params, posterior_precision=1000).detach().cpu().numpy().item()

      variance = log_p.var(dim=-1).mean().detach().cpu().numpy().item()
      s_value = results_laplace.query(f"model=='{labels[i]}'")["normalized KL"].item() * SUBSET_SIZE + expand_kl 
      s_value = (s_value + np.log(SUBSET_SIZE/delta)) / (SUBSET_SIZE - 1)
      s_values.append(s_value)
      # get item
      Iinv, lamb, J = rate_function_inv(log_p, s_value, device)
      
      expected_norm = compute_expected_norm(la).detach().cpu().numpy().item()
      expected_input_grad = compute_expected_input_gradient_norm(la, test_loader)
      

      inverse_rates.append(Iinv)
      variances.append(variance)
      lambdas.append(lamb)
      expected_cumulant.append(J)
      expected_norms.append(expected_norm)
      expected_input_grads.append(expected_input_grad)
      expanded_kl.append(expand_kl)
      t.update(1)

results_laplace["inverse rate"] = inverse_rates
results_laplace["s value"] = s_values
results_laplace["variance"] = variances
results_laplace["lambda"] = lambdas
results_laplace["expected cumulant"] = expected_cumulant
results_laplace["expected norm"] = expected_norms
results_laplace["expected input-gradient norm"] = expected_input_grads
results_laplace["expanded kl"] = expanded_kl


results_laplace.to_csv(csv_path, index=False)
print(results_laplace)
