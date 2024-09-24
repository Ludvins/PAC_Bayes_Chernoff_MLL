
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

from utils import latex_format, eval
import argparse
parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20
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
BATCH_SIZE = 200
BATCH_SIZE_TEST = 1000
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



subset = "last_layer"
hessian = "kron"

with tqdm(total=len(models)) as pbar:
  for name in models:
      pbar.set_description(f"Processing {name}")

      model = torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True)
      model = model.to(device)
      # Evaluate accuracy of the loaded model
      #accuracy = 0
      #accuracy = eval(device, model, test_loader, criterion)[0]
      #print(f'{name} accuracy: {accuracy:.2f}%')

    
      la = Laplace(model, "classification",
                    subset_of_weights=subset,
                    hessian_structure=hessian)
      la.fit(train_loader)
      la.optimize_prior_precision(prior_structure = args.prior_structure)
      torch.save(la.state_dict(), f'laplace_models_ResNet/{name}_{subset}_{hessian}_{args.prior_structure}_state_dict.pt')
      pbar.update(1)



