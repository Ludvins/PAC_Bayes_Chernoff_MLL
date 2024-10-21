#!/usr/bin/bash -l
#SBATCH --job-name=PAC_BAYES_DD
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arma@cs.aau.dk


nvidia-smi


BASEFOLDER=/home/cs.aau.dk/zp25hk/PAC_Bayes_Chernoff_MLL
PYTHON="singularity exec --nv $BASEFOLDER/containers/pytorch-3.10 python3"
SCRIPT=$BASEFOLDER/evaluate_laplace.py

$PYTHON $SCRIPT -m laplace_models -p 0.001 



