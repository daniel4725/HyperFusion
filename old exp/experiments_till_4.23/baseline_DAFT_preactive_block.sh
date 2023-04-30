#!/bin/bash
PYPROJ="/usr/local/data/danieldu/hyperproj/pyproj"
cd $PYPROJ

# args:
GPU="0"

exname="DAFT_preactive_block"
metadata_path="metadata_by_features_sets/set-5.csv"   # set 4 is norm minmax (0 to 1), set 5 is std-mean
model="DAFT_preactive_block"
cnn_dropout="0.1"
init_features="32"
lr="0.0001"
L2="0.00001"
epochs="300"
batch_size="64"
tform="hippo_crop_lNr"   # hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scale

adni_dir="/usr/local/faststorage/datasets/adni_class_pred_1x1x1_v1"

echo "starting cross validation exp..."
data_fold="0"
args="-exname $exname-f$data_fold --model $model  --cnn_dropout $cnn_dropout --init_features $init_features  -lr $lr --L2 $L2  --epochs $epochs --batch_size $batch_size --data_fold $data_fold -tform $tform --metadata_path $metadata_path --GPU $GPU -wandb -rfs -l2r --adni_dir $adni_dir -cp_dir $PYPROJ/checkpoints"
python3 model_trainer.py $args

data_fold="1"
args="-exname $exname-f$data_fold --model $model  --cnn_dropout $cnn_dropout --init_features $init_features  -lr $lr --L2 $L2  --epochs $epochs --batch_size $batch_size --data_fold $data_fold -tform $tform --metadata_path $metadata_path --GPU $GPU -wandb -rfs -l2r --adni_dir $adni_dir -cp_dir $PYPROJ/checkpoints"
python3 model_trainer.py $args

data_fold="2"
args="-exname $exname-f$data_fold --model $model  --cnn_dropout $cnn_dropout --init_features $init_features  -lr $lr --L2 $L2  --epochs $epochs --batch_size $batch_size --data_fold $data_fold -tform $tform --metadata_path $metadata_path --GPU $GPU -wandb -rfs -l2r --adni_dir $adni_dir -cp_dir $PYPROJ/checkpoints"
python3 model_trainer.py $args

data_fold="3"
args="-exname $exname-f$data_fold --model $model  --cnn_dropout $cnn_dropout --init_features $init_features  -lr $lr --L2 $L2  --epochs $epochs --batch_size $batch_size --data_fold $data_fold -tform $tform --metadata_path $metadata_path --GPU $GPU -wandb -rfs -l2r --adni_dir $adni_dir -cp_dir $PYPROJ/checkpoints"
python3 model_trainer.py $args

echo "Exiting..."
exit 0




