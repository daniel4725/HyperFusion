import os

base_cmd = f"sh {os.getcwd()}"


gpus = [2, 2, 2, 2]
# experiment: baseline-concatenation  baseline-tabular  DAFT  Film  TabularAsHyper  baseline-resnet  TabularAsHyper_trainedhyper  TabularAsHyper_trainedhyper8_cw085
# TabularAsHyper_cw085_2losses
# experiments = []
experiments = ["baseline-tabular", "TabularAsHyper_trainedhyper8_cw11d14",
               "baseline-concatenation_cw11_08_14", "DAFT_cw11d14", "Film_cw11d14"]
experiments = ["baseline-resnet"]
features_sets = [15]
seeds = [0, 1, 2]
versions = ["_v1", "_v2", "_v3"]

assert len(gpus) == 4
for gpu, fold in zip(gpus, [0, 1, 2, 3]):
    print("")
    for features_set in features_sets:
        for seed in seeds:
            for vers in versions:
                for experiment in experiments:
                    print(f"{base_cmd}/{experiment}.sh {gpu} {fold} {vers} {seed} {features_set}")

