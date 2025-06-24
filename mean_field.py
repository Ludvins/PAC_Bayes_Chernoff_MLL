"""Train MAP and MFVI models on CIFAR-10 and report
accuracy **and** negative log-likelihood (NLL).

Run e.g.:
    python cifar10_mfvi_nll.py --subset last_layer
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import copy
import pickle
import sys
from time import process_time as timer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from laplace import Laplace
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Local utilities
from utils import latex_format, eval  # noqa: E402
from bayesipy.mfvi import MFVI  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--subset", help="last_layer or all", type=str, default="all")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# General setup
# ---------------------------------------------------------------------------
latex_format()  # nicer matplotlib output if figures are produced

RANDOM_SEED = 2147483647
LEARNING_RATE = 0.01
SUBSET_SIZE = 50_000
TEST_SUBSET_SIZE = 10_000
N_ITERS = 2_000_000
BATCH_SIZE = 200
BATCH_SIZE_TEST = 1_000
IMG_SIZE = 32
N_CLASSES = 10
WEIGHT_DECAY = 0.01

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = datasets.CIFAR10(
    root="cifar_data", train=True, transform=transform, download=True
)
train_dataset = torch.utils.data.Subset(train_dataset, range(SUBSET_SIZE))


test_dataset = datasets.CIFAR10(
    root="cifar_data", train=False, transform=transform, download=True
)

test_dataset = torch.utils.data.Subset(test_dataset, range(TEST_SUBSET_SIZE))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
# CrossEntropyLoss with *sum* reduction lets us accumulate the total NLL and
# normalise once per dataset. For accuracy we just count correct predictions.
criterion_acc = nn.CrossEntropyLoss(reduction="none")
criterion_nll = nn.CrossEntropyLoss(reduction="sum")

def map_metrics(model):
    """Return accuracy and NLL of *deterministic* model on test set."""
    model.eval()
    correct, nll = 0, 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            # Accuracy
            correct += (logits.argmax(dim=1) == y).sum().item()
            # NLL (sum over batch)
            nll += criterion_nll(logits, y).item()
    n_samples = len(test_dataset)
    return correct / n_samples, nll / n_samples


def mfvi_metrics(mfvi_model):
    """Return accuracy and NLL of MFVI predictive distribution on test set."""
    correct, nll = 0, 0.0
    eps = 1e-12  # numerical stability for log
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits_samples = mfvi_model.predict(x)  # (S, B, C)
            probs = torch.softmax(logits_samples, dim=-1).mean(0)  # (B, C)
            preds = probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            # Gather probability of true class for each example
            p_true = probs[torch.arange(y.size(0)), y] + eps
            nll += (-p_true.log()).sum().item()
    n_samples = len(test_dataset)
    return correct / n_samples, nll / n_samples


def kl_mfvi_to_gaussian_prior(mfvi_model, prior_precision: float | torch.Tensor) -> torch.Tensor:

    mus, psis = [], []

    # 1. gather variational parameters -------------------------
    for name, p in mfvi_model.named_parameters():
        if name.endswith("mu"):
            mus.append(p)
        elif name.endswith("psi"):
            psis.append(p)

    if len(mus) != len(psis) or len(mus) == 0:
        raise RuntimeError("Mismatch: found %d mus but %d psis" % (len(mus), len(psis)))

    # 2. compute KL term --------------------------------------
    prior_var = 1.0 / prior_precision            # σ²ₚ
    kl_total  = torch.tensor(0.0, device=mus[0].device)

    for mu, psi in zip(mus, psis):
        var_q  = torch.exp(2.0 * psi)            # σ²_q  (because ψ = log σ)
        kl = 0.5 * torch.sum(
              torch.log(prior_var / var_q)       # log σ²ₚ / σ²_q
            + (var_q + mu.pow(2)) / prior_var
            - 1.0
        )
        kl_total = kl_total + kl

    return kl_total

# ---------------------------------------------------------------------------
# Main loop over stored checkpoints
# ---------------------------------------------------------------------------
model_type = "ConvNN"
labels = np.loadtxt(f"models/{model_type}_model_labels.txt", dtype=str)

with tqdm(total=len(labels)) as pbar:
    for label in labels:
        pbar.set_description(f"Processing {label}")

        # -------------------------------------------------------------------
        # Load deterministic (MAP) model
        # -------------------------------------------------------------------
        with open(f"models/{label}.pickle", "rb") as handle:
            map_model = pickle.load(handle).to(device)
        map_acc, map_nll = map_metrics(map_model)

        # -------------------------------------------------------------------
        # MFVI posterior around MAP
        # -------------------------------------------------------------------
        mfvi = MFVI(
            copy.deepcopy(map_model),
            n_samples=200,
            likelihood="classification",
            prior_precision=1.0,
            seed=RANDOM_SEED,
        )
        mfvi.fit(train_loader, 5_00, verbose=True)
        
        with open(f'mfvi_models/{label}.state_dict', 'wb') as handle:
            torch.save({'model': mfvi.state_dict()}, handle)
            
            
            
            
            
            
            
        with open(f'mfvi_models/{label}.state_dict', 'rb') as handle:
            checkpoint = torch.load(handle, map_location=device)   # {'model': …}
            
        mfvi = MFVI(
            copy.deepcopy(map_model),
            n_samples=200,
            likelihood="classification",
            prior_precision=1.0,
            seed=RANDOM_SEED,
        )
        mfvi.load_state_dict(checkpoint["model"])           
        mfvi_acc, mfvi_nll = mfvi_metrics(mfvi)
        
        KL = kl_mfvi_to_gaussian_prior(mfvi, prior_precision=1.0)
                
                
        print(
            f"{label} | MAP  acc={map_acc:.4f}, nll={map_nll:.4f}  | "
            f"MFVI acc={mfvi_acc:.4f}, nll={mfvi_nll:.4f}"
            f"  | KL={KL:.4f}"
        )

        pbar.update(1)