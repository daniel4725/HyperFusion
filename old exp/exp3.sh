#!/bin/bash

# TODO: change --time
#SBATCH --time=02:59:00

#SBATCH --account=rrg-arbeltal
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1 
#SBATCH --mem=46G

# TODO change name:
#SBATCH --output=%j-PreResNet_bn_3blks_drop03_64filt_BS64_croplNr.out
exname="PreResNet_bn_3blks_drop03_64filt_BS64_croplNr"
model="PreactivResNet_bn_3blks"
cnn_dropout="0.3"
init_features="64"
lr="0.0001"
L2="0.00001"
epochs="200"
batch_size="64"
tform="hippo_crop_lNr"   # hippo_crop  hippo_crop_lNr  normalize

# exname="PreResNet_Bn_5blks_drop02_32filt_BS6_x1"
# model="PreactivResNet_bn_5blks"
# cnn_dropout="0.2"
# init_features="32"
# lr="0.0001"
# L2="0.00001"
# epochs="200"
# batch_size="6"
# tform="normalize" 

GPU="0"
data_fold="0"

# ADNI_TYPE=adni_class_pred_2x2x2_v1
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
# python3 model_trainer.py -exname "PreResNet_Bn_5blks_drop02_32filt_BS6_x1" --GPU "0" --model "PreactivResNet_bn_5blks"  --cnn_dropout "0.2" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "200" --batch_size "6" --data_fold "0" -tform "normalize" -wandb -rfs --adni_dir $ADNI_DIR -cp_dir $PYPROJ/checkpoints 
# python3 model_trainer.py -exname "PreResNet_Bn_5blks_incDrop_32filt_BS6_x1" --GPU "0" --model "PreactivResNet_bn_5blks_incDrop"  --cnn_dropout "0.1" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "200" --batch_size "6" --data_fold "0" -tform "normalize" -wandb -rfs --adni_dir $ADNI_DIR -cp_dir $PYPROJ/checkpoints 
echo "Exiting..."
exit 0




