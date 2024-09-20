
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

from utils import latex_format, eval, eval_laplace, compute_trace

import argparse
parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-p", "--precision", help="Prior precision. If optimized, then -p=0", type=float, default=0)
parser.add_argument("-m", "--modelspath", help="Models folder name", default='./laplace_models')

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
                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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


models = ['cifar10_resnet20', 'cifar10_resnet32', 'cifar10_resnet44', 'cifar10_resnet56', 'cifar10_vgg11_bn', 
          'cifar10_vgg13_bn', 'cifar10_vgg16_bn', 'cifar10_vgg19_bn']

log_marginal = []
Gibbs_losses_train = []
Bayes_losses_train = []
Gibbs_losses = []
Bayes_losses = []
KLs = []
last_layer_params = []
prior_precisions = []
subset = "last_layer"
hessian = "kron"

with tqdm(total=len(models)) as pbar:
  for name in models:

      model = torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True)
      la = Laplace(model, "classification",
                  subset_of_weights=subset,
                  hessian_structure=hessian)
      la.load_state_dict(torch.load(f'{args.modelspath}/{name}_{subset}_{hessian}_state_dict.pt'))

      if float(args.precision) > 0:
          la.prior_precision = float(args.precision)
          prior = str(args.precision)
      else:
          prior = "opt"
          

      log_marginal.append(-la.log_marginal_likelihood(la.prior_precision).detach().cpu().numpy()/SUBSET_SIZE)
      
      trace_term, last_layer_param = compute_trace(la.posterior_precision)
      

      trace_term = la.prior_precision * trace_term
      kl = 0.5 * ( trace_term - last_layer_param + la.posterior_precision.logdet() - la.log_det_prior_precision + la.scatter)    
      
      last_layer_params.append(last_layer_param)
      KLs.append(kl.detach().cpu().numpy().item()/SUBSET_SIZE)
      bayes_loss, gibbs_loss = eval_laplace(device, la, test_loader)
      Bayes_losses.append(bayes_loss.detach().cpu().numpy())
      Gibbs_losses.append(gibbs_loss.detach().cpu().numpy())
      
      bayes_loss, gibbs_loss = eval_laplace(device, la, train_loader)
      Bayes_losses_train.append(bayes_loss.detach().cpu().numpy())
      Gibbs_losses_train.append(gibbs_loss.detach().cpu().numpy())
      prior_precisions.append(la.prior_precision.detach().cpu().numpy().item())
      pbar.set_description(f"Model {name}")
      pbar.update(1)

results = pd.DataFrame({'model': models, 
                        'subset': subset, 'hessian': hessian, 
                        "prior precision": prior_precisions, 
                       "bayes loss": Bayes_losses, 
                       "gibbs loss": Gibbs_losses, 
                       "bayes loss train": Bayes_losses_train,
                       "gibbs loss train": Gibbs_losses_train,
                       "neg log marginal laplace": log_marginal,
                       "neg log marginal": np.array(Gibbs_losses_train) + np.array(KLs),
                       "normalized KL": KLs,
                       "last layer params": last_layer_params})
results.to_csv(f"results/laplaceResNet_{subset}_{hessian}_"+prior+"_results.csv", index=False)
print(results)

