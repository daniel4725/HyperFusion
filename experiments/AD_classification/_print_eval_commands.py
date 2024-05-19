import os

gpu = 2

experiments = ["HyperFusion_ADvsCN_trained_tab"]

features_sets = [15]
seeds = [0, 1]
versions = ["_v1", "_v2"]

for features_set in features_sets:
    for seed in seeds:
        for vers in versions:
            for experiment in experiments:
                print(f"python3 {os.getcwd()}/_evaluate.py {gpu} {vers} {seed} {features_set} {experiment}")




