import shutil
import os
import cv2
import numpy as np
from tqdm import tqdm
import nibabel as nib
import threading
import time
import multiprocessing as mp
from torch import nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from data_handler import *
from tformNaugment import tform_dict
import pytorch_lightning as pl
from sklearn.manifold import TSNE

df = pd.read_csv('/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/my_adnimerege.csv')
df["Group"][(df["Group"] == "LMCI") | (df["Group"] == "EMCI")] = "MCI"

# ["AGE", "PTGENDER", "PTEDUCAT", "APOE4", # demographics and genetic Risk factors
# "ABETA", "PTAU", "TAU",  # CSF measures
# "FDG", "AV45"]   #  PET measures

# 185 nan in APOE4 and they intersect with all the rest of the Nans
# 1091 nan in ABETA, PTAU, TAU: 789 with AV45, 636 with FDG
# 1119 nan in AV45:  789 PTAU, TAU, ABETA.  651 with FDG
# 807 nan in FDG:  636 PTAU, TAU, ABETA.  651 with AV45

# non missing
have_csf_features = ~df["ABETA"].isna()
have_AV45 = ~df["AV45"].isna()
have_FDG = ~df["FDG"].isna()
for i in [have_csf_features, have_AV45, have_FDG, have_csf_features & have_FDG, have_csf_features & have_AV45, have_AV45 & have_FDG, have_FDG & have_csf_features & have_AV45]:
    # print("Total: ", i.sum())
    # print(df[i]["Group"].value_counts(normalize=True).round(3) * 100)
    print(df["Group"].value_counts())

# df[have_csf_features]["Group"].value_counts()
# df[have_AV45]["Group"].value_counts()
# df[have_FDG]["Group"].value_counts()
# df[have_csf_features & have_FDG]["Group"].value_counts()
# df[have_AV45 & have_FDG]["Group"].value_counts()
# df[have_csf_features & have_AV45]["Group"].value_counts()
# df[have_csf_features & have_AV45 & have_FDG]["Group"].value_counts()

df = df[have_AV45 & have_FDG & have_csf_features]
# Separate the data by groups
column = "Age"   # PTEDUCAT,  PTGENDER, Age, APOE4
ad_ages = df[df['Group'] == 'AD'][column]
mci_ages = df[df['Group'] == 'MCI'][column]
cn_ages = df[df['Group'] == 'CN'][column]

# Plot histograms
bins = 15
plt.hist(cn_ages, bins=bins, alpha=0.5, label='CN')
plt.hist(mci_ages, bins=bins, alpha=0.5, label='MCI')
plt.hist(ad_ages, bins=bins, alpha=0.5, label='AD')

# Add labels and legend
plt.xlabel(column)
plt.legend()
plt.show()
#
# # Separate the data by groups
# ad_ages = df[df['Group'] == 'AD']['Age']
# mci_ages = df[df['Group'] == 'MCI']['Age']
# cn_ages = df[df['Group'] == 'CN']['Age']
#
# # Plot histograms
# bins = 15
# plt.hist(cn_ages, bins=bins, alpha=0.5, label='CN')
# plt.hist(mci_ages, bins=bins, alpha=0.5, label='MCI')
# plt.hist(ad_ages, bins=bins, alpha=0.5, label='AD')
#
# # Add labels and legend
# plt.xlabel('Age')
# plt.legend()
# plt.show()
#
#
#
# # Separate the data by groups
# ad_ages_m = df[(df['Group'] == 'AD') & (df["PTGENDER"] == "Male")]['Age']
# ad_ages_f = df[(df['Group'] == 'AD') & (df["PTGENDER"] == "Female")]['Age']
# mci_ages_m = df[(df['Group'] == 'MCI') & (df["PTGENDER"] == "Male")]['Age']
# mci_ages_f = df[(df['Group'] == 'MCI') & (df["PTGENDER"] == "Female")]['Age']
# cn_ages_m = df[(df['Group'] == 'CN') & (df["PTGENDER"] == "Male")]['Age']
# cn_ages_f = df[(df['Group'] == 'CN') & (df["PTGENDER"] == "Female")]['Age']
#
#
# # Plot histograms
# bins = 15
# plt.hist(mci_ages_m, bins=bins, alpha=0.5, label='CN_male')
# plt.hist(mci_ages_f, bins=bins, alpha=0.5, label='CN_female')
#
# # Add labels and legend
# plt.xlabel('Age')
# plt.legend()
#
# # Show the plot
# plt.show()
