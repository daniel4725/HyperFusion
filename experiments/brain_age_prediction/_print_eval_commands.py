import os

gpu = 1

experiments = ["HyperFusion_Brainage"]


versions = ["_v1,_v2,_v3,_v4,_v5"]

for vers in versions:
    for experiment in experiments:
        print(f"python3 {os.getcwd()}/_evaluate.py {gpu} {vers} {experiment}")



