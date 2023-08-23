base_cmd = "sh /home/duenias/PycharmProjects/HyperNetworks/final_experiments_splitseed/"

# TODO  change class weights, cp path and project name in Tabular as hyper and in resnet baseine
# TODO   Tabularashyper epochs was 180,  resnet baseine epochs was 150
gpus = [0, 0, 0, 0]
# experiment: baseline-concatenation  baseline-tabular  DAFT  Film  TabularAsHyper  baseline-resnet  TabularAsHyper_trainedhyper  TabularAsHyper_trainedhyper8_cw085
# TabularAsHyper_cw085_2losses
# experiments = ["DAFT_cw11", "TabularAsHyper_trainedhyper8_cw09"]
# experiments = ["TabularAsHyper_trainedhyper8_cw11d10"]
experiments = ["baseline-resnet_cw11d14"]
features_sets = [15]
seeds = [0]
versions = ["_v1", "_v2", "_v3"]


assert len(gpus) == 4
for gpu, fold in zip(gpus, [0, 1, 2, 3]):
    print("")
    for features_set in features_sets:
        for seed in seeds:
            for vers in versions:
                for experiment in experiments:
                    print(f"{base_cmd}{experiment}.sh {gpu} {fold} {vers} {seed} {features_set}")


# ------------------ need to run ----------------
# Film & Daft diff weights (or hyper diff weights - same as Film & DAFT)
# concat diff weights? - same as DAFT or same as hyper?
# check out the v3 and v4 of Film (more epochs)
# More of fs5 for hyper & DAFT
# complete the commented experiments to get full 6 runs per experiment
#


