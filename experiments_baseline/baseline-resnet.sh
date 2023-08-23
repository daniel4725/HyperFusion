#!/bin/bash
PYPROJ="/home/duenias/PycharmProjects/HyperNetworks/pyproj"
cd $PYPROJ

# args:
GPU="$1"
data_fold="$2"
project_name="HyperNetworks_final"
#project_name="HyperNets_imgNtabular"

exname="$(basename "${0##*/}" | sed 's/\(.*\)\..*/\1/')"  # the experiment name is the file name
metadata_path="metadata_by_features_sets/set-5.csv"  # set 4 is norm minmax (0 to 1), set 5 is std-mean
model="PreactivResNet_bn_4blks_diff_start_incDrop_mlpend" #"PreactivResNet_bn_4blks_noDrop_mlpen"
cnn_dropout="0.1"
init_features="16"
lr="0.0001"
L2="0.00001"
epochs="140"
batch_size="64"
tform="hippo_crop_lNr_l2r"  # hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scale
tform_valid="None"  # hippo_crop_2sides hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scaletform_valid="hippo_crop_2sides"   # hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scale
num_workers="1"
class_weights="-cw 0.9454 1 1.9906"  # "-cw 0.9454 0.85 1.9906"  "-cw 0.9454 0.6945 1.9906"

# flags:
with_skull=""             # "--with_skull"  or ""
no_bias_field_correct="--no_bias_field_correct"   # "--no_bias_field_correct" or ""
load2ram="-l2r"            # "-l2r" or ""
ckpt_en="-ckpt_en"         # "-ckpt_en" or ""


adni_dir="/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI"
checkpoint_dir="/media/oldrrtammyfs/Users/daniel/HyperProj_checkpoints"

echo "starting cross validation exp..."
args="--project_name $project_name -exname $exname --model $model --init_features $init_features  -lr $lr --L2 $L2  --epochs $epochs --batch_size $batch_size --data_fold $data_fold -tform $tform --metadata_path $metadata_path --GPU $GPU -wandb -rfs --adni_dir $adni_dir -cp_dir $checkpoint_dir -nw $num_workers $with_skull $no_bias_field_correct $load2ram $ckpt_en $class_weights"
python3 model_trainer.py $args

#for data_fold in 3; do
#  args="-exname $exname --model $model --init_features $init_features  -lr $lr --L2 $L2  --epochs $epochs --batch_size $batch_size --data_fold $data_fold -tform $tform --metadata_path $metadata_path --GPU $GPU -wandb -rfs --adni_dir $adni_dir -cp_dir $checkpoint_dir -nw $num_workers $with_skull $no_bias_field_correct $load2ram $ckpt_en $class_weights"
#  python3 model_trainer.py $args
#done

echo "Exiting..."
exit 0
