import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression

features_sets = {}
# --------------- features_set 0 ------------------
set_num = 0
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER",] # demographics and genetic Risk factors
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median", "norm std-mean"],
    "PTGENDER": ["one_hot without_na"],
    }
# --------------- features_set 1 ------------------
set_num = 1
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["PTGENDER",] # demographics and genetic Risk factors
features_sets[set_num]["preprocess_dict"] = {
    "PTGENDER": ["one_hot without_na"],
    }

# --------------- features_set 2 ------------------
set_num = 2
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTEDUCAT", "APOE4",] # demographics and genetic Risk factors
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median", "norm std-mean"],
    "PTEDUCAT": ["norm std-mean"],
    "APOE4": ["norm std-mean", "add NaN col", "fill NaN with median"],
    }
# --------------- features_set 3 ------------------
set_num = 3
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4",] # demographics and genetic Risk factors
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median", "norm std-mean"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": ["norm std-mean"],
    "APOE4": ["one_hot with_na"],
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
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median", "norm std-mean"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": ["norm std-mean"],
    "APOE4": ["norm std-mean", "add NaN col", "fill NaN with median"],
    }
# --------------- features_set 9 ------------------
set_num = 9
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", # demographics and genetic Risk factors
                                      "ABETA", "PTAU", "TAU",  # CSF measures
                                      "FDG", "AV45"]   #  PET measures

features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median", "norm std-mean"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": ["norm std-mean"],
    "APOE4": ["add NaN col", "fill NaN with median", "norm std-mean"],
    "ABETA": ["remove><", "add NaN col", "fill NaN with median", "norm std-mean"],
    "PTAU": ["remove><", "add NaN col", "fill NaN with median", "norm std-mean"],
    "TAU": ["remove><", "add NaN col", "fill NaN with median", "norm std-mean"],
    "AV45": ["add NaN col", "fill NaN with median", "norm std-mean"],
    "FDG": ["add NaN col", "fill NaN with median", "norm std-mean"]
    }
# --------------- features_set 10 ------------------
set_num = 10
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", # demographics and genetic Risk factors
                                      "ABETA", "PTAU", "TAU",  # CSF measures
                                      "FDG", "AV45"]   #  PET measures

features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": [],
    "APOE4": [],
    "ABETA": ["remove><"],
    "PTAU": ["remove><"],
    "TAU": ["remove><"],
    "AV45": [],
    "FDG": [],
    "all_together": ["impute_all Nan_col", "normalize_all"]
    }
# --------------- features_set 11 ------------------
set_num = 11
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", # demographics and genetic Risk factors
                                      "ABETA", "PTAU", "TAU",  # CSF measures
                                      "FDG", "AV45"]   #  PET measures

features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": [],
    "APOE4": [],
    "ABETA": ["remove><"],
    "PTAU": ["remove><"],
    "TAU": ["remove><"],
    "AV45": [],
    "FDG": [],
    "all_together": ["impute_all", "normalize_all"]
    }
# --------------- features_set 12 ------------------
set_num = 12
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", # demographics and genetic Risk factors
                                      "ABETA", "PTAU", "TAU",  # CSF measures
                                      "FDG", "AV45"]   #  PET measures
features_sets[set_num]["preprocess_dict"] = {
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
# --------------- features_set 13 ------------------
set_num = 13
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", # demographics and genetic Risk factors
                                      "ABETA", "PTAU", "TAU",  # CSF measures
                                      "FDG", "AV45"]   #  PET measures
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": [],
    "APOE4": [],
    "ABETA": ["remove><"],
    "PTAU": ["remove><"],
    "TAU": ["remove><"],
    "AV45": [],
    "FDG": [],
    "all_together": ["impute_all", "normalize_all_but_Na"]
    }
# --------------- features_set 14 ------------------
set_num = 14
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", # demographics and genetic Risk factors
                                      "ABETA", "PTAU", "TAU",  # CSF measures
                                      "FDG", "AV45"]   #  PET measures
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median", "norm std-mean"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": ["norm std-mean"],
    "APOE4": ["fill NaN with median", "norm std-mean"],
    "ABETA": ["remove><", "norm std-mean"],
    "PTAU": ["remove><", "norm std-mean"],
    "TAU": ["remove><", "norm std-mean"],
    "AV45": ["norm std-mean"],
    "FDG": ["norm std-mean"],
    "all_together": ["fill_NaN_by_group"]
    }
# --------------- features_set 15 ------------------
set_num = 15
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", # demographics and genetic Risk factors
                                      "ABETA", "PTAU", "TAU",  # CSF measures
                                      "FDG", "AV45"]   #  PET measures
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": [],
    "APOE4": [],
    "ABETA": ["remove><"],
    "PTAU": ["remove><"],
    "TAU": ["remove><"],
    "AV45": [],
    "FDG": [],
    "all_together": ["impute_all Nan_col", "normalize_all_but_Na&Gender"]
    }
# --------------- features_set 16 ------------------
set_num = 16
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE"]
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median"],
    }
# --------------- features_set 17 ------------------
set_num = 17
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER"]
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median"],
    "PTGENDER": ["one_hot without_na"],
    }
# --------------- features_set 18 ------------------
set_num = 18
features_sets[set_num] = {}
features_sets[set_num]["features"] = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4"]
features_sets[set_num]["preprocess_dict"] = {
    "AGE": ["fill NaN with median"],
    "PTGENDER": ["one_hot without_na"],
    "PTEDUCAT": [],
    "APOE4": [],
    "all_together": ["impute_all Nan_col", "normalize_all_but_Na&Gender"]
    }


def gauss_1d(n=10,sigma=1, mu=None):
    if mu is None:
        mu = n//2
    r = range(n)
    gaussian = np.array([1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x - mu)**2/(2*sigma**2)) for x in r])
    gaussian[gaussian < 1e-8] = 0
    gaussian /= gaussian.sum()  # normalize to sum to 1
    return gaussian

def preprocess_df_columns(csv, col_namesNoperations_dict: dict, split_seed=0, fold=0):
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
            elif operation == "fill NaN with median w.r.t labels":
                for label in csv["Group"].unique():
                    labels_median = csv[col_name][csv["Group"] == label].median()
                    csv[col_name][csv[col_name].isna() & (csv["Group"] == label)] = labels_median
            elif operation == "fill NaN with mean":
                csv[col_name].fillna(csv[col_name].mean(), inplace=True)

            elif operation == "remove><":
                for i in range(len(csv[col_name])):
                    if csv[col_name][i] is not np.nan:
                        if (csv[col_name][i][0] == '<') or (csv[col_name][i][0] == '>'):
                            csv.loc[i, col_name] = csv[col_name][i][1:]
                    if csv[col_name][i] is not np.nan:
                        csv.loc[i, col_name] = float(csv[col_name][i])

            elif ("impute_all" in operation) and col_name == "all_together":
                add_indicator = True if "Nan_col" in operation else False
                imp = IterativeImputer(max_iter=200, initial_strategy="median", random_state=0, add_indicator=add_indicator)
                cols = list(csv.columns)
                cols.remove("Subject")
                cols.remove("Group")
                imputed_csv = imp.fit_transform(csv[cols])
                new_cols = [col for col in cols]
                if add_indicator:
                    for col in cols:
                        if csv[col].isna().sum() > 0:
                            new_cols.append(f"{col}_Na")
                df = pd.DataFrame(data=imputed_csv, columns=new_cols)
                df[["Group", "Subject"]] = csv[["Group", "Subject"]]
                csv = df


            elif operation == "normalize_all" and col_name == "all_together":
                cols = list(csv.columns)
                cols.remove("Subject")
                cols.remove("Group")
                csv[cols] = (csv[cols] - csv[cols].mean()) / csv[cols].std()

            elif operation == "normalize_all_but_Na" and col_name == "all_together":
                cols = [c for c in csv.columns if "Na" not in c]
                cols.remove("Subject")
                cols.remove("Group")
                csv[cols] = (csv[cols] - csv[cols].mean()) / csv[cols].std()

            elif operation == "normalize_all_but_Na&Gender" and col_name == "all_together":
                cols = [c for c in csv.columns if "Na" not in c]
                cols.remove("Subject")
                cols.remove("Group")
                cols.remove("PTGENDER_Male")
                cols.remove("PTGENDER_Female")
                csv[cols] = (csv[cols] - csv[cols].mean()) / csv[cols].std()



            elif operation == "delete_nan":
                csv = csv[~csv[col_name].isna()]
                csv.reset_index(drop=True, inplace=True)

            elif operation == "fill_NaN_by_group":
                skf = StratifiedKFold(n_splits=5, random_state=split_seed, shuffle=True)
                X = csv.drop(['Subject', 'Group'], axis=1)
                y = csv["Group"]
                list_of_splits = list(skf.split(X, y))
                _, val_idxs = list_of_splits[fold]
                _, test_idxs = list_of_splits[4]
                train_idxs = list(np.where(~csv.index.isin(list(val_idxs) + list(test_idxs)))[0])

                # impute the training set with the help of all the values (including the group)
                tmp = csv.copy()
                tmp = pd.get_dummies(tmp, dummy_na=False, columns=["Group"])
                cols =list(tmp.columns)
                cols.remove("Subject")
                imp = IterativeImputer(max_iter=200, initial_strategy="median", random_state=0, add_indicator=False)
                tmp.loc[train_idxs, cols] = pd.DataFrame(data=imp.fit_transform(tmp.loc[train_idxs, cols]), columns=cols, index=train_idxs)
                cols =list(csv.columns)
                cols.remove("Group")
                cols.remove("Subject")
                csv.loc[train_idxs, cols] = tmp.loc[train_idxs, cols]

                # impute the test and validation set without the group column
                cols =list(csv.columns)
                cols.remove("Group")
                cols.remove("Subject")
                imp = IterativeImputer(max_iter=200, initial_strategy="median", random_state=0, add_indicator=False)
                csv.loc[val_idxs, cols] = pd.DataFrame(data=imp.fit_transform(csv.loc[val_idxs, cols]), columns=cols, index=val_idxs)
                imp = IterativeImputer(max_iter=200, initial_strategy="median", random_state=0, add_indicator=False)
                csv.loc[test_idxs, cols] = pd.DataFrame(data=imp.fit_transform(csv.loc[test_idxs, cols]), columns=cols, index=test_idxs)

            else:
                raise AssertionError(f"operation '{operation}' for column '{col_name}' in preprocess_columns function is'nt from the optional actions")
    return csv


def create_metadata_csv(features_set_idx, csv_path="/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/my_adnimerege.csv",
                        split_seed=0, fold=0):
    global features_sets

    features_lst = features_sets[features_set_idx]["features"]
    features_preprocess_dict = features_sets[features_set_idx]["preprocess_dict"]

    adni_csv = pd.read_csv(csv_path)

    features_lst = features_lst + ['Subject', 'Group']  # add the target and the Subject id
    adni_csv = adni_csv.loc[:, features_lst]

    adni_csv = preprocess_df_columns(adni_csv, features_preprocess_dict, split_seed, fold)

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
    
    features_set_idx = 2

    ADNI_dir = f"/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023"
    metadata_dir = "metadata_by_features_sets"
    save_metadata_path = os.path.join(metadata_dir, f"set-{features_set_idx}.csv")

    create_metadata_csv(features_set_idx=features_set_idx)

