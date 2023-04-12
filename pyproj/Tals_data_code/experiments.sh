# ------- 19.10.22 ---------
# simple first tst:
python model_trainer.py -exname "tst2" --GPU "2" -lr "0.0001"  --epochs "3" -cp_en -wandb -l2r -rfs

# ------- 19.10.22 ---------
# small hyper search
python model_trainer.py -exname "Res" --GPU "2" --model "ResNet" --init_features "4" -lr "0.0001" --L2 "0.1"  --epochs "3" --batch_size "8" --data_fold "0" -cp_en -wandb -l2r -rfs

# ------- 30.10.22 ---------
# best hyper parameters for ResNet alone: L2=0.01 batch_size=8 init_features=8 lr=0.0001

python model_trainer.py -exname "ResNet_0" --GPU "2" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "0" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "ResNet_1" --GPU "2" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "1" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "ResNet_2" --GPU "2" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "2" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "ResNet_3" --GPU "3" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "3" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "ResNet_4" --GPU "3" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "4" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"

python model_trainer.py -exname "MLP4Tabular_0" --GPU "1" --model "MLP4Tabular" --hidden_shapes "16" "16" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "0" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "MLP4Tabular_1" --GPU "1" --model "MLP4Tabular" --hidden_shapes "16" "16" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "1" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "MLP4Tabular_2" --GPU "1" --model "MLP4Tabular" --hidden_shapes "16" "16" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "2" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "MLP4Tabular_3" --GPU "1" --model "MLP4Tabular" --hidden_shapes "16" "16" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "3" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "MLP4Tabular_4" --GPU "1" --model "MLP4Tabular" --hidden_shapes "16" "16" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "4" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"

python model_trainer.py -exname "ResNetHyperEnd_0" --GPU "2" --model "ResNetHyperEnd" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "0" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "ResNetHyperEnd_1" --GPU "2" --model "ResNetHyperEnd" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "1" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "ResNetHyperEnd_2" --GPU "3" --model "ResNetHyperEnd" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "2" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "ResNetHyperEnd_3" --GPU "3" --model "ResNetHyperEnd" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "3" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"
python model_trainer.py -exname "ResNetHyperEnd_4" --GPU "3" --model "ResNetHyperEnd" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "4" -wandb -l2r -rfs --metadata_path "metadata_features_set_2.csv"


python model_trainer.py -exname "ResNet_0" --GPU "2" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "0" -tform "normalize" -wandb -l2r -rfs
python model_trainer.py -exname "ResNet_1" --GPU "2" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "1" -tform "augment2" -wandb -l2r -rfs
python model_trainer.py -exname "ResNet_2" --GPU "3" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "2" -tform "augment2" -wandb -l2r -rfs
python model_trainer.py -exname "ResNet_3" --GPU "3" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "70" --batch_size "8" --data_fold "3" -tform "augment2" -wandb -l2r -rfs


python model_trainer.py -exname "ResNet_0_notform" --GPU "2" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "20" --batch_size "8" --data_fold "0" -tform "None" -wandb -l2r -rfs
python model_trainer.py -exname "ResNet_0_aug1" --GPU "3" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "20" --batch_size "8" --data_fold "0" -tform "augment1" -wandb -l2r -rfs


python model_trainer.py -exname "ResNet_0_base" --GPU "2" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "8" --data_fold "0" -tform "normalize" -wandb -l2r -rfs
python model_trainer.py -exname "ResNet2_0_base" --GPU "3" --model "ResNet2" --cnn_mlp_shapes "8" "3" --cnn_mlp_dropout "0.0" --cnn_dropout "0.0" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "8" --data_fold "0" -tform "normalize" -wandb -l2r -rfs
python model_trainer.py -exname "ResNet2_0_drop" --GPU "2" --model "ResNet2" --cnn_mlp_shapes "16" "3" --cnn_mlp_dropout "0.1" --cnn_dropout "0.1" --init_features "16" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "8" --data_fold "0" -tform "normalize" -wandb -l2r -rfs


python model_trainer.py -exname "ResNet2_0_base_aug" --GPU "3" --model "ResNet2" --cnn_mlp_shapes "8" "3" --cnn_mlp_dropout "0.0" --cnn_dropout "0.0" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "8" --data_fold "0" -tform "basic_aug" -wandb -l2r -rfs
python model_trainer.py -exname "ResNet2_0_drop_aug" --GPU "2" --model "ResNet2" --cnn_mlp_shapes "16" "3" --cnn_mlp_dropout "0.1" --cnn_dropout "0.1" --init_features "16" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "8" --data_fold "0" -tform "basic_aug" -wandb -l2r -rfs


python model_trainer.py -exname "ResNet_0_base_aug" --GPU "2" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "8" --data_fold "0" -tform "basic_aug" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-1x1x1_set-1.csv"
python model_trainer.py -exname "ResNet2_0_base_aug" --GPU "3" --model "ResNet2" --cnn_mlp_shapes "8" "3" --cnn_mlp_dropout "0.0" --cnn_dropout "0.0" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "8" --data_fold "0" -tform "basic_aug" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-1x1x1_set-1.csv"


python model_trainer.py -exname "ResNet_0_base_aug1" --GPU "2" --model "ResNet" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "8" --data_fold "0" -tform "basic_aug" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-1x1x1_set-1.csv"
python model_trainer.py -exname "ResNet2_0_base_aug1" --GPU "3" --model "ResNet2" --cnn_mlp_shapes "8" "3" --cnn_mlp_dropout "0.0" --cnn_dropout "0.0" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "8" --data_fold "0" -tform "basic_aug" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-1x1x1_set-1.csv"

# 2 classes trials
python model_trainer.py -exname "ResNet_CN_AD_norm" --GPU "2" --model "ResNet" --class_names "CN" "AD" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "10" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-1x1x1_set-1.csv"
python model_trainer.py -exname "ResNet2_CN_AD_norm" --GPU "3" --model "ResNet2" --class_names "CN" "AD" --cnn_mlp_shapes "8" "2" --cnn_mlp_dropout "0.0" --cnn_dropout "0.0" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "10" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-1x1x1_set-1.csv"

# with some dropout
python model_trainer.py -exname "ResNet2_CN_AD_norm" --GPU "2" --model "ResNet2" --class_names "CN" "AD" --cnn_mlp_shapes "8" "2" --cnn_mlp_dropout "0.05" --cnn_dropout "0.05" --init_features "8" -lr "0.0001" --L2 "0.03"  --epochs "40" --batch_size "10" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-1x1x1_set-1.csv"

# test VGG style network
python model_trainer.py -exname "BasicCNN_norm" --GPU "2" --model "BasicCNN" --class_names "CN" "MCI" "AD"  --init_features "4" -lr "0.0001" --L2 "0.03"  --epochs "40" --batch_size "8" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "BasicCNN_basic_aug" --GPU "3" --model "BasicCNN" --class_names "CN" "MCI" "AD"  --init_features "4" -lr "0.0001" --L2 "0.03"  --epochs "40" --batch_size "8" --data_fold "0" -tform "basic_aug" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "BasicCNN_norm_cw02" --GPU "2" --model "BasicCNN" --class_names "CN" "MCI" "AD"  --init_features "4" -lr "0.0001" --L2 "0.03"  --epochs "40" --batch_size "8" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"

# test MCI weight reduced
python model_trainer.py -exname "ResNet_norm" --GPU "3" --model "ResNet" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "16" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "ResNet_norm_cw03" --GPU "3" --model "ResNet" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "16" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"

python model_trainer.py -exname "ResNet_norm_yesCW" --GPU "2" --model "ResNet" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "16" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "ResNet_norm_yesCW_LS" --GPU "2" --model "ResNet" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "16" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "ResNet_norm_noCW_LS" --GPU "0" --model "ResNet" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "16" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"


python model_trainer.py -exname "ResNetInstnorm_norm_noCW" --GPU "0" --model "ResNetInstnorm" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "16" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "ResNetInstnorm_norm_noCW_BS2" --GPU "2" --model "ResNetInstnorm" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "2" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"

# TODO run with affine=True
python model_trainer.py -exname "ResNetInstnorm_norm_noCW_BS2_affine" --GPU "1" --model "ResNetInstnorm" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.01"  --epochs "40" --batch_size "2" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"



python model_trainer.py -exname "ResNet_norm_noCW_L2small" --GPU "0" --model "ResNet" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.00001"  --epochs "40" --batch_size "64" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"

python model_trainer.py -exname "ResNet_norm_noCW_L2small_drop" --GPU "0" --model "ResNetDrop" --class_names "CN" "MCI" "AD" --init_features "8" -lr "0.0001" --L2 "0.00001"  --epochs "40" --batch_size "64" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "ResNet_norm_noCW_Bn_drop02_32filt" --GPU "0" --model "ResNetDrop" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "40" --batch_size "64" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"


# preactivation resnet
python model_trainer.py -exname "PreResNet_noCW_Bn_drop02_32filt" --GPU "3" --model "PreactivResNet_bn" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "40" --batch_size "32" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "PreResNet_noCW_Bn_drop03_32filt" --GPU "1" --model "PreactivResNet_bn" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "40" --batch_size "32" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "PreResNet_noCW_instN_drop02_32filt" --GPU "2" --model "PreactivResNet_instN" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "40" --batch_size "32" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "PreResNet_noCW_instN_drop02_32filt_BS2_1x1x1" --GPU "2" --model "PreactivResNet_instN" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "40" --batch_size "4" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-1x1x1_set-1.csv"


python model_trainer.py -exname "PreResNet_yesCW_Bn_drop02_32filt" --GPU "3" --model "PreactivResNet_bn" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "40" --batch_size "32" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "PreResNet_noCW_instN_drop0_32filt_BS2_1x1x1" --GPU "3" --model "PreactivResNet_instN" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "40" --batch_size "4" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-1x1x1_set-1.csv"


# --------------------------------------------------------------------------------
# binary classifications:
python model_trainer.py -exname "PreResNet_CN-AD_Bn_drop03" --GPU "3" --model "PreactivResNet_bn" --class_names "CN" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "60" --batch_size "32" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "PreResNet_CN-MCI_Bn_drop03" --GPU "3" --model "PreactivResNet_bn" --class_names "CN" "MCI" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "60" --batch_size "32" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "PreResNet_MCI-AD_Bn_drop03" --GPU "1" --model "PreactivResNet_bn" --class_names "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "60" --batch_size "32" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"


python model_trainer.py -exname "PreResNet_drop04_32filt" --GPU "2" --model "PreactivResNet_bn" --cnn_dropout "0.4" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "80" --batch_size "32" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv"
python model_trainer.py -exname "PreResNet_drop05_32filt" --GPU "3" --model "PreactivResNet_bn" --cnn_dropout "0.5" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "80" --batch_size "32" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --metadata_path "metadata_by_features_sets/images-2x2x2_set-1.csv" 

python model_trainer.py -exname "PreResNet_drop04_BS4_1x1x1" --GPU "2" --model "PreactivResNet_bn" --cnn_dropout "0.4" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "80" --batch_size "4" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --adni_dir "/usr/local/faststorage/adni_class_pred_1x1x1_v1"


python model_trainer.py -exname "PreResNet_instN_drop0_32filt_BS2_1x1x1" --GPU "1" --model "PreactivResNet_instN"  --cnn_dropout "0.0" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "256" --batch_size "4" --data_fold "0" -tform "normalize" -wandb -l2r -rfs --adni_dir "/usr/local/faststorage/adni_class_pred_1x1x1_v1"


# hippo crop:
python model_trainer.py -exname "PreResNet_instN_drop01_32filt_BS32_x1croped" --GPU "0" --model "PreactivResNet_instN"  --cnn_dropout "0.1" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "256" --batch_size "32" --data_fold "0" -tform "hippo_crop" -wandb -rfs --adni_dir "/usr/local/faststorage/adni_class_pred_1x1x1_v1"
python model_trainer.py -exname "PreResNet_instN_drop03_32filt_BS32_x1croped" --GPU "2" --model "PreactivResNet_instN"  --cnn_dropout "0.3" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "256" --batch_size "32" --data_fold "0" -tform "hippo_crop" -wandb -rfs --adni_dir "/usr/local/faststorage/adni_class_pred_1x1x1_v1"
python model_trainer.py -exname "PreRNet_instN_3blks_drop02_64filt_BS64_x1croped" --GPU "0" --model "PreactivResNet_instN_3blks"  --cnn_dropout "0.2" --class_names "CN" "MCI" "AD" --init_features "64" -lr "0.0001" --L2 "0.00001"  --epochs "256" --batch_size "64" --data_fold "0" -tform "hippo_crop" -wandb -rfs --adni_dir "/usr/local/faststorage/adni_class_pred_1x1x1_v1"
python model_trainer.py -exname "PreRNet_Bn_3blks_drop04_64filt_BS64_x1croped" --GPU "3" --model "PreactivResNet_bn_3blks"  --cnn_dropout "0.4" --class_names "CN" "MCI" "AD" --init_features "64" -lr "0.0001" --L2 "0.00001"  --epochs "256" --batch_size "64" --data_fold "0" -tform "hippo_crop" -wandb -rfs --adni_dir "/usr/local/faststorage/adni_class_pred_1x1x1_v1"

# 5blks
python model_trainer.py -exname "PreResNet_instN_5blks_drop01_32filt_BS4_x1" --GPU "2" --model "PreactivResNet_instN_5blks"  --cnn_dropout "0.1" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "256" --batch_size "4" --data_fold "0" -tform "normalize" -wandb -rfs --adni_dir "/usr/local/faststorage/adni_class_pred_1x1x1_v1"

python3 model_trainer.py -exname "PreResNet_Bn_5blks_drop01_32filt_BS8_x1" --GPU "0" --model "PreactivResNet_bn_5blks"  --cnn_dropout "0.1" --class_names "CN" "MCI" "AD" --init_features "32" -lr "0.0001" --L2 "0.00001"  --epochs "200" --batch_size "8" --data_fold "0" -tform "normalize" -wandb -rfs --adni_dir "/usr/local/faststorage/adni_class_pred_1x1x1_v1"

