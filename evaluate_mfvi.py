# evaluate_mfvi_checkpoints.py
# ---------------------------------------------------------------
import glob, os, copy, pickle, argparse, math
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from bayesipy.mfvi import MFVI                      
from utils import eval as eval_utils, latex_format  
from utils import rate_function_inv_mfvi, get_log_p_mfvi

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



def mfvi_metrics(mfvi_model, loader):
    """
    Return
        bayes_loss    – negative log-likelihood under Bayesian model average
        gibbs_loss    – expected NLL of a single Gibbs posterior sample
        bma_acc       – accuracy of the Bayesian model average
        gibbs_acc     – mode-of-samples (Gibbs) accuracy
    """
    bayes_loss  = 0.0
    gibbs_loss  = 0.0
    bma_correct = 0
    gibbs_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += y.size(0)

            # (S, B, C) logits samples
            logits_samples = mfvi_model.predict(x)          # your MFVI method
            S = logits_samples.size(0)

            # one-hot targets and per-sample log-prob of true class (no softmax)
            oh = F.one_hot(y, num_classes=logits_samples.size(-1))
            log_prob = (logits_samples * oh).sum(-1) \
                       - torch.logsumexp(logits_samples, dim=-1)

            # ---------- losses ----------
            #  \log(1/S Σ e^{log_prob})  done in a stable log-space form
            bayes_loss -= torch.logsumexp(
                log_prob - math.log(S), dim=0
            ).sum()

            #  mean over samples first, then log
            gibbs_loss -= log_prob.mean(0).sum()

            # ---------- accuracies ----------
            # Bayesian model average
            avg_logits = logits_samples.mean(0)              # (B, C)
            bma_correct += (avg_logits.argmax(-1) == y).sum().item()

            # Gibbs / majority vote of S draws
            sample_preds = logits_samples.argmax(-1)         # (S, B)
            gibbs_pred, _ = torch.mode(sample_preds, dim=0)  # (B,)
            gibbs_correct += (gibbs_pred == y).sum().item()
 
    return (
        bayes_loss / total,
        gibbs_loss / total,
        bma_correct / total,
        gibbs_correct / total,
    )



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




# ----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="mfvi_models", help="where *.state_dict live")
parser.add_argument("--csv", default="metrics.csv", help="output csv")
parser.add_argument("--prior", default=1.0, type=float, help="prior precision")
args = parser.parse_args()

# --------------------------- Data --------------------------------
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)]
)

train_dataset = torch.utils.data.Subset(
    datasets.CIFAR10("cifar_data", train=True, transform=transform, download=True),
    range(5_000),
)
train_loader = DataLoader(train_dataset, batch_size=2_00, shuffle=False)


test_dataset = torch.utils.data.Subset(
    datasets.CIFAR10("cifar_data", train=False, transform=transform, download=True),
    range(5_000),
)
test_loader = DataLoader(test_dataset, batch_size=2_00, shuffle=False)

# --------------------------- loop --------------------------------

labels = np.loadtxt(f"models/ConvNN_model_labels.txt", delimiter=" ", dtype = str)
n_params = np.loadtxt(f"models/ConvNN_n_params.txt")

records=[]


with tqdm(range(len(n_params))) as bar:
    for i in range(len(n_params)):
        label = labels[i]
        bar.set_description(label)

        # ---------------------------------------------------------
        # 1. reconstruct the deterministic (MAP) network
        #    You saved it earlier as `<label>.pickle`
        # ---------------------------------------------------------
        with open(f"models/{label}.pickle", "rb") as fh:
            map_model = pickle.load(fh).to(device)
        map_acc, map_nll = map_metrics(map_model)

        # ---------------------------------------------------------
        # 2. create an *empty* MFVI wrapper and load state_dict
        # ---------------------------------------------------------
        mfvi = MFVI(
            copy.deepcopy(map_model),
            n_samples=64,
            likelihood="classification",
            prior_precision=args.prior,
            seed=RANDOM_SEED,
        )
        ckpt = torch.load(f'{args.dir}/{labels[i]}_{args.prior}.state_dict', map_location=device)
        mfvi.load_state_dict(ckpt["model"])

        bayes_loss_test, gibbs_loss_test, bma_acc_test, gibbs_acc_test  = mfvi_metrics(mfvi, test_loader)
        bayes_loss_train, gibbs_loss_train, bma_acc_train, gibbs_acc_train = mfvi_metrics(mfvi, train_loader)
        kl = kl_mfvi_to_gaussian_prior(mfvi, prior_precision=args.prior).item()
        s_value = (kl + np.log(SUBSET_SIZE/0.01))/(SUBSET_SIZE)
        Iinv, lamb, J = rate_function_inv_mfvi(mfvi, test_loader, s_value, device)

        # ---------------------------------------------------------
        records.append(
            dict(label=label,
                 map_acc_test=map_acc,      
                 map_nll_test=map_nll,
                 bayes_loss_test=bayes_loss_test.cpu().numpy().item(),
                 gibbs_loss_test=gibbs_loss_test.cpu().numpy().item(),
                 bayes_loss_train=bayes_loss_train.cpu().numpy().item(),
                 gibbs_loss_train=gibbs_loss_train.cpu().numpy().item(),
                 bma_acc_test=bma_acc_test,
                 bma_acc_train=bma_acc_train,
                 kl=kl,
                 inverse_rate=Iinv.cpu().numpy().item()
                 )
        )
        bar.update(1)

# --------------------------- CSV ---------------------------------
df = pd.DataFrame.from_records(records)
df.to_csv(f"results/ConvNN_mfvi_{args.prior}.csv", index=False)
print(f"\nSaved metrics for {len(df)} checkpoints {args.csv}")
