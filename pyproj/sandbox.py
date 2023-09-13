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
from utils import *

df = pd.read_csv("/media/rrtammyfs/labDatabase/BrainAge/Healthy_subjects_divided_pipe_v2.csv", low_memory=False)
projects = df.ProjTitle.unique()

for proj in projects:
    subset = df[df.ProjTitle == proj]
    if "Oasis" in proj:
        subset = df[(df.ProjTitle == "OasisCross") | (df.ProjTitle == "OasisLong")]
    if "ADNI" in proj:
        subset = df[(df.ProjTitle == "ADNIDOD") | (df.ProjTitle == "ADNI")]

    subset = subset[~(subset.Age.isna() | subset.Gender.isna())]
    N = len(subset)
    age = subset.Age
    males = (subset.Gender == "M").sum()
    females = (subset.Gender == "F").sum()
    print(f"{proj} & {N} & {age.mean():.1f} (±{age.std():.1f}) & {males}:{females} \\\\ \\hline")

subset = df[~(df.Age.isna() | df.Gender.isna())]
N = len(subset)
age = subset.Age
males = (subset.Gender == "M").sum()
females = (subset.Gender == "F").sum()
print(f"over all & {N} & {age.mean():.1f} (±{age.std():.1f}) & {males}:{females} \\\\ \\hline")

