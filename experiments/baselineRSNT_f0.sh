#!/bin/bash

# TODO: change --time
#SBATCH --time=02:59:00

#SBATCH --account=rrg-arbeltal
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1 
#SBATCH --mem=46G

# TODO change name:
#SBATCH --output=%j-mlpend[06d64,05d3].out
exname="mlpend[06d64,05d3]_f0"
model="PreactivResNet_bn_4blks_incDrop_mlpend"
cnn_dropout="0.1"
init_features="32"
lr="0.0001"
L2="0.00001"
epochs="300"
batch_size="64"
tform="hippo_crop_lNr"   # hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scale

GPU="0"
data_fold="0"

ADNI_TYPE="adni_class_pred_1x1x1_v1"

PYPROJ=$HOME/projects/def-arbeltal/$USER/pyproj

# NOTE: This data will be uncompressed directly on to the compute node
DATA=$HOME/projects/def-arbeltal/data/tars/$ADNI_TYPE.tar

echo "Untaring data..."
tar -xf $DATA -C $SLURM_TMPDIR
ADNI_DIR=$SLURM_TMPDIR/$ADNI_TYPE

args="-exname $exname --model $model  --cnn_dropout $cnn_dropout --init_features $init_features  -lr $lr --L2 $L2  --epochs $epochs --batch_size $batch_size --data_fold $data_fold -tform $tform --GPU $GPU -wandb -rfs --adni_dir $ADNI_DIR -cp_dir $PYPROJ/checkpoints"
echo "Experiment args:"
echo $args

echo "Loading modules"
module load python/3.8.10
module load httpproxy

echo "source virtual environment..."
source $HOME/venv/bin/activate


echo "Beginning Experiment..."
cd $PYPROJ
wandb login --relogin 588ccf0f1631d995e385af0b2675f0214d2fb9ff
wandb online

python3 model_trainer.py $args
echo "Exiting..."
exit 0




