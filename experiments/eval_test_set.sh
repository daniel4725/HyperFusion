#!/bin/bash
PYPROJ="/home/duenias/PycharmProjects/HyperNetworks/pyproj"
cd $PYPROJ

# args:
#project_name="HyperNetworks_final"
project_name="HyperNets_imgNtabular"

GPU="1"
exname="baseline-resnet-fs8-c3"  # the experiment name is the file name
features_set="8"  # set 4 is norm minmax (0 to 1), set 5 is std-mean
num_classes="3"

batch_size="64"
tform="hippo_crop_lNr"  #  hippo_crop_lNr  hippo_crop_lNr_l2r  ----normalize hippo_crop_lNr_noise hippo_crop_lNr_scale
tform_valid="hippo_crop_2sides"  # hippo_crop_2sides  None ----hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scaletform_valid="hippo_crop_2sides"   # hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scale
num_workers="7"

# flags:
with_skull=""             # "--with_skull"  or ""
no_bias_field_correct="--no_bias_field_correct"   # "--no_bias_field_correct" or ""
load2ram=""            # "-l2r" or ""


adni_dir="/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI"
checkpoint_dir="/media/oldrrtammyfs/Users/daniel/HyperProj_checkpoints"

echo "starting test set evaluation..."
args="--project_name $project_name -exname $exname  --batch_size $batch_size  --features_set $features_set --GPU $GPU -wandb -rfs --adni_dir $adni_dir -cp_dir $checkpoint_dir -nw $num_workers $with_skull $no_bias_field_correct $load2ram --num_classes $num_classes"
python3 model_test_evaluator.py $args

echo "Exiting..."
exit 0
