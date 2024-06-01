import os

gpus = [2, 2, 3, 3]

# experiments = ["baseline-tabular"]
# experiments = ["baseline-imaging"]
# experiments = ["baseline-concatenation"]
# experiments = ["HyperFusion_AD"]
# experiments = ["DAFT"]
# experiments = ["FiLM"]
# experiments = ["HyperFusion_ablation_ADvsCN"]
# experiments = ["HyperFusion_AD_hyper_first_block"]
# experiments = ["HyperFusion_AD_hyper_sec_block"]
experiments = ["HyperFusion_AD_hyper_3rd_block"]
# experiments = ["HyperFusion_AD_hyper_2fc"]
# experiments = ["HyperFusion_AD_hyper_TFF", "HyperFusion_AD_hyper_2fc", "HyperFusion_AD_hyper_sec_block"]

features_sets = [15]
seeds = [0, 1]
versions = ["_v1", "_v2"]

assert len(gpus) == 4
for gpu, fold in zip(gpus, [0, 1, 2, 3]):
    print("")
    for features_set in features_sets:
        for seed in seeds:
            for vers in versions:
                for experiment in experiments:
                    print(f"python3 {os.getcwd()}/{experiment}.py {gpu} {fold} {vers} {seed} {features_set}")
