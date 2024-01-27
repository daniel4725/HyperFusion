import os

gpus = [2, 2, 3, 3]

experiments = ["baseline-tabular", "TabularAsHyper_trainedhyper8_cw11d14",
               "baseline-concatenation_cw11_08_14", "DAFT_cw11d14", "Film_cw11d14"]

# experiments = ["baseline-tabular"]
experiments = ["baseline-imaging"]
# experiments = ["baseline-concatenation"]
# experiments = ["HyperFusion"]
# experiments = ["DAFT"]
# experiments = ["FiLM"]

features_sets = [15]
seeds = [0]
versions = ["_v1"]

assert len(gpus) == 4
for gpu, fold in zip(gpus, [0, 1, 2, 3]):
    print("")
    for features_set in features_sets:
        for seed in seeds:
            for vers in versions:
                for experiment in experiments:
                    print(f"python3 {os.getcwd()}/{experiment}.py {gpu} {fold} {vers} {seed} {features_set}")

