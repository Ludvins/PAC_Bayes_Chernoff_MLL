
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import numpy as np
import torchvision
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from laplace import Laplace
from tqdm import tqdm
import subprocess

from utils import latex_format, eval, eval_laplace, compute_trace

import argparse
parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-p", "--precision", help="Prior precision. If optimized, then -p=0", type=float)
parser.add_argument("-m", "--modelspath", help="Models folder name")
parser.add_argument("-ps", "--prior_structure", help="scalar, layerwise or diag", type=str)
parser.add_argument("--subset", help = "last_layer or all", type=str)
args = parser.parse_args()



# Activate Latex format for matplotlib
latex_format()

# Create custom loss functions
criterion = nn.CrossEntropyLoss() # supervised classification loss

# Hyper-Parameters
RANDOM_SEED = 2147483647
LEARNING_RATE = 0.01
SUBSET_SIZE = 50000
TEST_SUBSET_SIZE = 10000
N_ITERS = 2000000
BATCH_SIZE = 32
BATCH_SIZE_TEST = 32
IMG_SIZE = 32
N_CLASSES = 10
WEIGHT_DECAY = 0.01


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

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE_TEST,
                          shuffle=False)


#######################################################################
############################# TRAIN MODELS ############################
#######################################################################


model_type = "ConvNN"
labels = np.loadtxt(f"models/{model_type}_model_labels.txt", delimiter=" ", dtype = str)
n_params = np.loadtxt(f"models/{model_type}_n_params.txt")

log_marginal = []
Gibbs_losses_train = []
Bayes_losses_train = []
Gibbs_losses = []
Bayes_losses = []
BMA_test_accuracy = []
BMA_train_accuracy = []
KLs = []
last_layer_params = []
prior_precisions = []

hessian = "Kron"

with tqdm(range(len(n_params))) as t:
  for i in range(len(n_params)):

    with open(f"models/{labels[i]}.pickle", "rb") as handle:
    
      model = pickle.load(handle)

      la = Laplace(model, "classification",
                  subset_of_weights=args.subset,
                  hessian_structure=hessian)
       
      la.load_state_dict(torch.load(f'{args.modelspath}/{labels[i]}_{args.subset}_{hessian}_{args.prior_structure}_state_dict.pt'))

      if args.prior_structure == "scalar" and float(args.precision) > 0:
          la.prior_precision = float(args.precision)
          prior = str(args.precision)
      else:
          prior = "opt " + args.prior_structure
          
      print(f"Prior precision: {la.prior_precision}")
      log_marginal.append(-la.log_marginal_likelihood(la.prior_precision).detach().cpu().numpy()/SUBSET_SIZE)
      
      trace_term, last_layer_param = compute_trace(la.posterior_precision)
      

      trace_term = la.prior_precision * trace_term
      kl = 0.5 * ( trace_term - last_layer_param + la.posterior_precision.logdet() - la.log_det_prior_precision + la.scatter)    
      
      last_layer_params.append(last_layer_param)
      KLs.append(kl.detach().cpu().numpy().item()/SUBSET_SIZE)
      bayes_loss, gibbs_loss, bma = eval_laplace(device, la, test_loader)
      Bayes_losses.append(bayes_loss.detach().cpu().numpy())
      Gibbs_losses.append(gibbs_loss.detach().cpu().numpy())
      BMA_test_accuracy.append(bma)

      bayes_loss, gibbs_loss, bma = eval_laplace(device, la, train_loader)
      Bayes_losses_train.append(bayes_loss.detach().cpu().numpy())
      Gibbs_losses_train.append(gibbs_loss.detach().cpu().numpy())
      BMA_train_accuracy.append(bma)


      if args.prior_structure == "scalar":
        prior_precisions.append(la.prior_precision.detach().cpu().numpy().item())
      else:
        prior_precisions.append(prior)
      
      del la, model, bayes_loss, gibbs_loss, bma, kl, last_layer_param
      torch.cuda.empty_cache()  

      t.set_description(f"Model {labels[i]}")
      t.update(1)

results = pd.DataFrame({'model': labels, 'parameters': n_params, 
                        'subset': args.subset, 'hessian': hessian, 
                        "prior precision": prior_precisions, 
                        "BMA test accuracy (%)": BMA_test_accuracy,
                        "BMA train accuracy (%)": BMA_train_accuracy,
                        "bayes loss": Bayes_losses,
                        "gibbs loss": Gibbs_losses, 
                        "bayes loss train": Bayes_losses_train,
                        "gibbs loss train": Gibbs_losses_train,
                        "neg log marginal laplace": log_marginal,
                       "neg log marginal": np.array(Gibbs_losses_train) + np.array(KLs),
                       "normalized KL": KLs,
                       "last layer params": last_layer_params
                       })
results.to_csv(f"results/laplace_{model_type}_{args.subset}_{hessian}_{args.prior_structure}_{args.precision}_results.csv", index=False)
print(results)

