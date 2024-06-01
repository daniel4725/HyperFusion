import os

gpu = 1

# experiments = ["HyperFusion_ADvsCN_trained_tab"]
# experiments = ["Hyper_first_block"]
# experiments = ["HyperFusion_AD_hyper_second_block"]
experiments = ["HyperFusion_AD_hyper_3rd_block"]
# experiments = ["HyperFusion_AD_TFF"]
# experiments = ["HyperFusion_AD_hyper_2fc"]


features_sets = [15]
seeds = [0, 1]
versions = ["_v1", "_v2"]

for features_set in features_sets:
    for seed in seeds:
        for vers in versions:
            for experiment in experiments:
                print(f"python3 {os.getcwd()}/_evaluate.py {gpu} {vers} {seed} {features_set} {experiment}")



