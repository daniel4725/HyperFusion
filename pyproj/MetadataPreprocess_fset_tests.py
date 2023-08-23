"""
CHALLANGE SUGGEST:
The main measures to be predicted: DX, ADAS13, Ventricles
Cognitive tests: CDRSB, ADAS11, MMSE, RAVLT_immediate
MRI measures: Hippocampus, WholeBrain, Entorhinal, MidTemp
PET measures: FDG, AV45
CSF measures: ABETA  (amyloid-beta level in CSF), TAU (tau level T-tau), PTAU (phosphorylated tau level P-tau181)
Risk factors: APOE4, AGE

Kirill using:
So in my case, I used brain volume measures (Hippocampus, WholeBrain, Entorhinal, MidTemp, Ventricles),
cognitive test scores (CDRSB, ADAS13, ADAS11, MMSE, RAVLT_immediate) and APOE4, along with disease 
stage diagnosis at baseline (DX_bl)

DAFT using:
Tabular data comprises 9 variables: age (AGE), gender (PTGENDER), education (PTEDUCAT), ApoE4 (APOE4), cerebrospinal
fluid biomarkers Aβ42 (ABETA), P-tau181 (PTAU) and T-tau (TAU), and two summary measures derived
from 18F-fluorodeoxyglucose (FDG) and florbetapir (AV45) PET scans.

"""

"""
All relevant features:
        ["AGE", "PTGENDER", "PTEDUCAT", "APOE4",  # demographics and genetic Risk factors
        PTETHCAT - Ethnicity
        PTRACCAT - Race
        PTMARRY - ['Married', 'Divorced', 'Widowed', 'Never married', 'Unknown']
         "ABETA", "PTAU", "TAU",   # CSF measures
         "FDG", "AV45",  # PET measures
         "Hippocampus", "WholeBrain", "Entorhinal", "MidTemp", "Ventricles",  # MRI measures
         "CDRSB", "ADAS13", "ADAS11", "MMSE", "RAVLT_immediate",  # Cognitive tests
         "DX_bl",  # Target
         'IMAGEUID']  # the image id

    "AGE":"norm min-max",
    "PTGENDER":"one_hot with_na",
    "PTEDUCAT":"",
    "APOE4":"",
    "ABETA":"",
    "PTAU":"",
    "TAU":"",
    "Hippocampus":"",
    "WholeBrain":"",
    "Entorhinal":"",
    "MidTemp":"",
    "Ventricles":"",
    "CDRSB":"",
    "ADAS13":"",
    "ADAS11":"",
    "MMSE":"",
    "RAVLT_immediate":"",

"""
import itertools

all_features = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4",  # demographics and genetic Risk factors
                "ABETA", "PTAU", "TAU",  # CSF measures
                "FDG", "AV45"]  # PET measures
operations_dict = {
    "AGE": ["fill NaN with median"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": [],
    "APOE4": [],
    "ABETA": ["remove><"],
    "PTAU": ["remove><"],
    "TAU": ["remove><"],
    "AV45": [],
    "FDG": [],
    "all_together": ["impute_all Nan_col", "normalize_all_but_Na"]
}

def add_features_sets(features_sets: dict):
    global all_features, operations_dict
    base_set_num = 100
    combnations = list(itertools.combinations(all_features, 3))
    # combnations = []

    set_num = base_set_num
    for comb in combnations:
        print(list(comb))
        features_sets[set_num] = {}
        features_sets[set_num]["features"] = list(comb)
        features_sets[set_num]["preprocess_dict"] = {}
        for feature in comb:
            features_sets[set_num]["preprocess_dict"][feature] = operations_dict[feature]
        features_sets[set_num]["preprocess_dict"]["all_together"] = ["impute_all Nan_col", "normalize_all_but_Na"]
        set_num += 1


if __name__ == "__main__":
    
    f = {}
    add_features_sets(f)

    a = 5