#!/bin/bash

project_name="HyperNetworks_final_splitseed_test"
#project_name="HyperNetworks_final_splitseed"
#project_name="HyperNets_imgNtabular"

GPU="3"
split_seed="1"
version="_v3"   # the base experiment
versions=$version
#versions="_v1,_v2,_v3"   # the ensemble experiments
features_set="18"
# TabularAsHyper_R_R_R_FFT_FF_embd8_cw1  DAFT_BalancCw  Film_cw  baseline-concat1_cw1  baseline-tabular_embd8

#experiment_base_name="DAFT_cw11d14"
#experiment_base_name="Film_cw11d14"
#experiment_base_name="baseline-concat1_cw11_08_14"
#experiment_base_name="TabularAsHyper_embd_trainedTabular8_cw11d14"
experiment_base_name="baseline-tabular_embd8"


#experiment_base_name="Film_cw11d14"
#experiment_base_name="baseline-concat1_cw11_08_14"
#experiment_base_name="baseline-resnet_cw1"
#experiment_base_name="T_AsHyper_R_R_FFT_FFF_FF_embd_trainedTabular8_cw085"
#experiment_base_name="TabularAsHyper_scaled_R_R_R_FFT_FF_embd_trainedTabular_cw085"
#experiment_base_name="ImageAsHyper_Hembd16_FT"
#experiment_base_name="T_AsHyper_R_R_FFT_FFT_FF_cw075085_embd_trainedTabular8"
#experiment_base_name="TabularAsHyper_embd_trainedTabular8_cw11085185"
#experiment_base_name="TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw109085"
#experiment_base_name="TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw104085"
#experiment_base_name="TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw09"
#experiment_base_name="TabularAsHyper_embd_trainedTabular8_cwdd17"
#experiment_base_name="DAFT_BalancCw"
#experiment_base_name="DAFT_cw085"
#experiment_base_name="DAFT_cwdd17"
#experiment_base_name="DAFT_cw09"
#experiment_base_name="DAFT_cw11085185"
#experiment_base_name="DAFT_defaultCW"
#experiment_base_name="TabularAsHyper_embd_trainedTabular_defaultCW"
#experiment_base_name="TabularAsHyper_embd_trainedTabular8_cw1118"


experiment_name="$experiment_base_name$version-seed$split_seed-fs$features_set"
eval_script_path="/home/duenias/PycharmProjects/HyperNetworks/final_experiments_splitseed/_evaluate_on_test_set.sh"
sh "$eval_script_path" $GPU $project_name $experiment_name $features_set $split_seed $versions
