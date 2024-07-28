import os


# experiments = ["experiments_sandbox"]
experiments = ["HyperFusion_brainage"]
# experiments = ["baseline-concatenation"]
# experiments = ["baseline-imaging"]
# experiments = ["HyperFusion_brainage_tests"]

# experiments = ["B_fill__FT_FFF"]
# experiments = ["B_nofill__TTTT"]
# experiments = ["B_fill__FFFT", "B_fill__FFTF", "B_fill__FTFF", "B_fill__TFFF"]
# experiments = ["B_nofill__TTTT", "B_fill__FFFT", "B_nofill__FFFT", "B_nofill__FT_FFFF"]


gpu = 2
versions = ["_v1"]

for vers in versions:
    for experiment in experiments:
        print(f"python3 {os.getcwd()}/{experiment}.py {gpu} {vers}")

