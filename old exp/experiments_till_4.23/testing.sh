#!/bin/bash
PYPROJ="/home/duenias/PycharmProjects/HyperNetworks/pyproj"
cd $PYPROJ

# args:
GPU="1"

exname="testing"
metadata_path="metadata_by_features_sets/set-5.csv"   # set 4 is norm minmax (0 to 1), set 5 is std-mean
model="age_noembd_lastblockHyp_FFT_fcHyp_2"
cnn_dropout="0.1"
init_features="32"
lr="0.0001"
L2="0.00001"
epochs="350"
batch_size="64"
tform="hippo_crop_lNr"   # hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scale

adni_dir="/media/rrtammyfs/Users/daniel/adni_class_pred_1x1x1_v1"

echo "starting cross validation exp..."
data_fold="0"
args="-exname $exname-f$data_fold --model $model  --cnn_dropout $cnn_dropout --init_features $init_features  -lr $lr --L2 $L2  --epochs $epochs --batch_size $batch_size --data_fold $data_fold -tform $tform --metadata_path $metadata_path --GPU $GPU  -rfs -l2r -wandb --adni_dir $adni_dir -cp_dir $PYPROJ/checkpoints  -nw 0"
python3.6 model_trainer.py $args

echo "Exiting..."
exit 0




