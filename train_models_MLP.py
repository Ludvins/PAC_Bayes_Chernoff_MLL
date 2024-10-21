
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import numpy as np
import torchvision
import pickle

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from utils import latex_format, MLPcreatemodel, train

# Activate Latex format for matplotlib
latex_format()

# Create custom loss functions
criterion = nn.CrossEntropyLoss() # supervised classification loss

# Hyper-Parameters
RANDOM_SEED = 2147483647
LEARNING_RATE = 0.01
SUBSET_SIZE = 50000
TEST_SUBSET_SIZE = 10000
N_ITERS = 10000
BATCH_SIZE = 200
BATCH_SIZE_TEST = 1000
INPUT_SHAPE = (1,28,28)
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

transform = transforms.Compose([
    				transforms.ToTensor(), 
    				transforms.Normalize((0.5,), (0.5,)),])  

train_dataset = datasets.MNIST(root='./data', 
				train=True, 
				download=True, 
				transform=transform)

#######################################################################
############################# TRAIN MODELS ############################
#######################################################################


# Initialize models
models = [MLPcreatemodel(RANDOM_SEED, INPUT_SHAPE, [k, k], N_CLASSES).to(device) for k in [10, 20, 40, 80, 160, 320, 640, 1280, 2560]]

n_params = []
for model in models:
  n = 0
  for parameter in model.parameters():
    n += parameter.flatten().size(0)
  n_params.append(n)


labels = ["MLP-"+str(p//1000)+"k" for p in n_params]

np.savetxt("models/MLP_model_labels.txt",labels, delimiter=" ", fmt="%s")
np.savetxt("models/MLP_n_params.txt",n_params)


for i in range(len(models)):
  name = labels[i]
  g_cuda = torch.Generator(device='cpu')
  g_cuda.manual_seed(RANDOM_SEED)
  loader = torch.utils.data.DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          generator=g_cuda,
                          shuffle=True)
  train(models[i], name, loader, LEARNING_RATE, N_ITERS, device, criterion)
  
  with open(f'models/{labels[i]}.pickle', 'wb') as handle:
    pickle.dump(models[i], handle, protocol=pickle.HIGHEST_PROTOCOL)

