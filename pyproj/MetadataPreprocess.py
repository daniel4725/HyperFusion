#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
# adni_csv.describe()

features_sets = {}

# --------------- features_set 1 ------------------
set_num = 1
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", # demographics and genetic Risk factors
                            "CDRSB", "ADAS13", "ADAS11", "MMSE", "RAVLT_immediate"]  # Cognitive tests
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["norm min-max"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": ["norm min-max"],
    "APOE4": ["add NaN col", "fill NaN with median", "norm min-max"],
    "CDRSB": ["norm min-max"],
    "ADAS13": ["add NaN col", "fill NaN with median", "norm min-max"],
    "ADAS11": ["add NaN col", "fill NaN with median", "norm min-max"],
    "MMSE": ["norm min-max"],
    "RAVLT_immediate": ["add NaN col", "fill NaN with median", "norm min-max"]
}

# --------------- features_set 2 ------------------
set_num = 2
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4"] # demographics and genetic Risk factors
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["norm min-max"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": ["norm min-max"],
    "APOE4": ["add NaN col", "fill NaN with median", "norm min-max"]
}

# --------------- features_set 3 ------------------
set_num = 3
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4"] # demographics and genetic Risk factors
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["norm min-max"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": ["norm min-max"],
    "APOE4": ["add NaN col", "fill NaN with median", "norm min-max"],
    "ABETA": ["add NaN col", "fill NaN with median", "norm min-max"],
    "PTAU": ["add NaN col", "fill NaN with median", "norm min-max"],
    "TAU": ["add NaN col", "fill NaN with median", "norm min-max"],
    # "APOE4": ["add NaN col", "fill NaN with median", "norm min-max"],
    # "APOE4": ["add NaN col", "fill NaN with median", "norm min-max"]

# and two summary measures derived
# from 18F-fluorodeoxyglucose (FDG) and florbetapir (AV45) PET scans.

}
# --------------- features_set 4 ------------------
set_num = 4
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE"]
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["norm min-max"]
}
# --------------- features_set 5 ------------------
set_num = 5
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE"]
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median", "norm std-mean"]
}
# --------------- features_set 6 ------------------
set_num = 6
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE"]
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["AGE_smoothed_one_hot"]
}
# --------------- features_set 7 ------------------
set_num = 7
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE"]
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["AGE_binary_cutoff_69"]
}
# --------------- features_set 8 ------------------
set_num = 8
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4",] # demographics and genetic Risk factors
                                    # "ABETA", "PTAU", "TAU",  # CSF measures
                                    # "FDG", "AV45"]   #  PET measures 

features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median", "norm std-mean"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": ["norm std-mean"],
    "APOE4": ["norm std-mean", "add NaN col", "fill NaN with median"],
    # "ABETA": ["remove><", "add NaN col", "fill NaN with median", "norm std-mean"],
    # "PTAU": ["remove><", "add NaN col", "fill NaN with median", "norm std-mean"],
    # "TAU": ["remove><", "add NaN col", "fill NaN with median", "norm std-mean"],
    # "AV45": [],   451 missing values out of 864
    # "FDG": ["add NaN col", "fill NaN with median", "norm std-mean"]
    }
# --------------------------------------------------

def gauss_1d(n=10,sigma=1, mu=None):
    if mu is None:
        mu = n//2
    r = range(n)
    gaussian = np.array([1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x - mu)**2/(2*sigma**2)) for x in r])
    gaussian[gaussian < 1e-8] = 0
    gaussian /= gaussian.sum()  # normalize to sum to 1
    return gaussian

def preprocess_df_columns(csv, col_namesNoperations_dict: dict):
    """ col_namesNoperations_dict is an ordered dict of operations
    to do at each column in the following form:
    {col_name1:[op1,op2,op3..] , col_name2:[op1, ...]} """

    for col_name in col_namesNoperations_dict.keys():
        operations = col_namesNoperations_dict[col_name]
        for operation in operations:
            if operation == "one_hot with_na":
                csv = pd.get_dummies(csv, dummy_na=True, columns=[col_name])
            elif operation == "one_hot without_na":
                csv = pd.get_dummies(csv, dummy_na=False, columns=[col_name])

            elif operation == "AGE_smoothed_one_hot":
                n = int(csv["AGE"].max() - csv["AGE"].min()) + 1
                smoothed_ages = []
                for age in csv["AGE"]:
                    smoothed_ages.append(gauss_1d(n=n, sigma=1, mu=age - csv["AGE"].min()))
                # csv[col_name] = csv[col_name].round()
                cols = ["AGE_{}".format(i + int(csv["AGE"].min())) for i in range(n)]
                csv[cols] = smoothed_ages
                csv.drop('AGE', inplace=True, axis=1)

            elif operation == "AGE_binary_cutoff_69":
                csv["AGE"][csv["AGE"] <= 69] = 0
                csv["AGE"][csv["AGE"] > 69] = 1

            elif operation == "norm min-max":
                csv[col_name] = (csv[col_name] - csv[col_name].min())/(csv[col_name].max() - csv[col_name].min())
            elif operation == "norm std-mean":
                csv[col_name] = (csv[col_name] - csv[col_name].mean())/csv[col_name].std()
            
            elif operation == "add NaN col":  # adds a column named col_name_NaN with 1 in each NaN row of col_name
                csv[col_name + '_nan'] = csv[col_name].isna().astype('int')
            
            elif operation == "fill NaN with median":
                csv[col_name].fillna(csv[col_name].median(), inplace=True)
            elif operation == "fill NaN with mean":
                csv[col_name].fillna(csv[col_name].mean(), inplace=True)

            elif operation == "remove><":
                for i in range(len(csv[col_name])):
                    if csv[col_name][i] is not np.nan:
                        if (csv[col_name][i][0] == '<') or (csv[col_name][i][0] == '>'):
                            csv[col_name][i] = csv[col_name][i][1:]
                    if csv[col_name][i] is not np.nan:
                        csv[col_name][i] = float(csv[col_name][i])

            
            else:
                raise ValueError(f"operation '{operation}' for column '{col_name}' in preprocess_columns function is'nt from the optional actions")
    return csv


def create_metadata_csv(ADNI_dir, save_path, features_set_idx):
    global features_sets

    ########################################################################
    # csv = pd.read_csv("/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/my_adnimerege.csv")
    # for img_id in csv["Image Data ID"]:
    #     print(img_id + ",", end="")

    # dates_diff_threshold = 50  # naximum days differances between the scan and the other tabular data collection
    # dates_diff_mask = ((csv["dates_diff_mergecsv_mri_scan"] < dates_diff_threshold) & (csv["dates_diff_demogcsv_mri_scan"] < dates_diff_threshold))
    # csv = csv[dates_diff_mask]
    ########################################################################

    features_lst = features_sets[features_set_idx]["features"]
    features_preprocess_dict = features_sets[features_set_idx]["preprocess_dict"]

    adni_csv = pd.read_csv(os.path.join(ADNI_dir, "my_adnimerege.csv"))

    features_lst = features_lst + ['Subject', 'Group']  # add the target and the Subject id
    adni_csv = adni_csv.loc[:, features_lst]
    adni_csv.loc[adni_csv["Group"] == "CN", "Group"] = 0
    adni_csv.loc[adni_csv["Group"] == "LMCI", "Group"] = 1
    adni_csv.loc[adni_csv["Group"] == "EMCI", "Group"] = 1
    adni_csv.loc[adni_csv["Group"] == "MCI", "Group"] = 1
    adni_csv.loc[adni_csv["Group"] == "AD", "Group"] = 2

    adni_csv = preprocess_df_columns(adni_csv, features_preprocess_dict) 

    adni_csv.to_csv(save_path, index=False)
    return adni_csv
  
def feature_properties(df_col):
    # feature_properties(adni_csv["CDRSB"])
    print(f"column name: {df_col.name}")
    print(f"dtype: {df_col.dtype}")
    print(f"unique vals: {pd.unique(df_col)}")
    print(f"len: {len(df_col)}")
    print(f"num nan: {df_col.isna().sum()}")
    df_col.hist()
    plt.show()


if __name__ == "__main__":
    
    features_set_idx = 5

    ADNI_dir = f"/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023"
    metadata_dir = "metadata_by_features_sets"
    save_metadata_path = os.path.join(metadata_dir, f"set-{features_set_idx}.csv")

    create_metadata_csv(ADNI_dir=ADNI_dir, save_path=save_metadata_path, features_set_idx=features_set_idx)
