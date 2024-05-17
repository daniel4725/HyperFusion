import os


# experiments = ["experiments_sandbox"]
# experiments = ["HyperFusion_brainage"]
# experiments = ["baseline-concatenation"]
experiments = ["baseline-imaging"]
# experiments = ["HyperFusion_brainage_tests"]


gpu = 0
versions = ["_v5", "_v6", "_v7", "_v8", "_v9", "_v10"]

for vers in versions:
    for experiment in experiments:
        print(f"python3 {os.getcwd()}/{experiment}.py {gpu} {vers}")


