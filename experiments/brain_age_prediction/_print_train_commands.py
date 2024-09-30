import os


# experiments = ["experiments_sandbox"]
# experiments = ["HyperFusion_brainage"]
experiments = ["baseline-concatenation"]
# experiments = ["baseline-imaging"]
# experiments = ["HyperFusion_brainage_tests"]


gpu = 2
versions = ["_v1"]

for vers in versions:
    for experiment in experiments:
        print(f"python3 {os.getcwd()}/{experiment}.py {gpu} {vers}")

