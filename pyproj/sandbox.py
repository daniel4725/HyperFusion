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

df = pd.read_csv('/media/rrtammyfs/Users/daniel/hyper_table_val_aranged.csv', index_col=0)
df = df.head()

title = "Title"
metrics = ['balanced_acc', 'precision', 'f1_macro', 'AUC', 'CN_acc', 'MCI_acc', 'AD_acc']
metrics = ['balanced_acc', 'precision', 'f1_macro']
colors = []
colors += ['tab:blue']
colors += ['tab:cyan']
colors += ['tab:olive']
colors += ['tab:red']
colors += ['tab:purple']
colors += ['tab:orange']
colors += ['tab:green']
colors += ['tab:brown']
colors += ['tab:pink']
colors += ['tab:grey']


results_df2bars_graph(df, title, metrics, colors)