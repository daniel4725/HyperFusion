#!/bin/bash
PYPROJ="/home/duenias/PycharmProjects/HyperNetworks/pyproj"
cd $PYPROJ

GPU="2"
project_name="HyperNetworks_final"
#project_name="HyperNets_imgNtabular"
exname="TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_aa-fs5"  # the experiment name is the file name
features_set="5"
batch_size="16"


num_classes=3
tform="hippo_crop_lNr"  # hippo_crop_lNr_l2r  hippo_crop_lNr         normalize hippo_crop_lNr_noise hippo_crop_lNr_scale
tform_valid="hippo_crop_2sides"  # None hippo_crop_2sides           hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scaletform_valid="hippo_crop_2sides"   # hippo_crop  hippo_crop_lNr  normalize hippo_crop_lNr_noise hippo_crop_lNr_scale
num_workers="15"
# flags:
with_skull=""             # "--with_skull"  or ""
no_bias_field_correct="--no_bias_field_correct"   # "--no_bias_field_correct" or ""
load2ram=""            # "-l2r" or ""


adni_dir="/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI"
checkpoint_dir="/media/oldrrtammyfs/Users/daniel/HyperProj_checkpoints"

#exname="$exname-fs$features_set"
args="--project_name $project_name -exname $exname --batch_size $batch_size -tform $tform --features_set $features_set --GPU $GPU -wandb -rfs --adni_dir $adni_dir -cp_dir $checkpoint_dir -nw $num_workers $with_skull $no_bias_field_correct $load2ram $ckpt_en $class_weights --num_classes $num_classes"
python3 model_test_evaluator.py $args


echo "Exiting..."
exit 0
