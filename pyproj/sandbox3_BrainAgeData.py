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
from transformers import AutoModel, AutoConfig


def cpy_images2server():
    dataset_dir = "/media/rrtammyfs/labDatabase/BrainAge/Healthy/"
    destenation_base = "/home/duenias/PycharmProjects/HyperNetworks/BrainAgeDataset/brain_scans"
    subjects = os.listdir(dataset_dir)
    # copy all the np simple images to the server
    for subj in tqdm(subjects):
        src_path = os.path.join(dataset_dir, subj, "numpySave_simple", f"{subj}.npy")
        dst_dir = os.path.join(destenation_base, subj)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, f"brain_scan.npy")
        shutil.copyfile(src_path, dst_path)

        # img = np.load(src_path)
        try:
            np.load(dst_path)
        except:
            print(f"copy error in subject: {subj}")


def copy_metadata2server():
    csv_path = "/media/rrtammyfs/labDatabase/BrainAge/Healthy_subjects_divided_pipe_v2.csv"
    csv_dst_path = "/home/duenias/PycharmProjects/HyperNetworks/BrainAgeDataset/metadata.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[["Subject", "Age", "Gender"]]
    df = df[~(df["Age"].isna() | df["Gender"].isna() | (df["Gender"] == "X"))]
    df.to_csv(csv_dst_path, index=False)


def check_all_images():
    dataset_dir = "/media/rrtammyfs/labDatabase/BrainAge/Healthy/"
    destenation_base = "/home/duenias/PycharmProjects/HyperNetworks/BrainAgeDataset/brain_scans"
    subjects = os.listdir(dataset_dir)
    for subj in tqdm(subjects):
        dst_dir = os.path.join(destenation_base, subj)
        dst_path = os.path.join(dst_dir, f"brain_scan.npy")
        try:
            np.load(dst_path)
        except:
            print(f"copy error in subject: {subj}")


def delete_non_relevant_scans():
    data_dir = "/home/duenias/PycharmProjects/HyperNetworks/BrainAgeDataset/brain_scans"
    csv = pd.read_csv("/home/duenias/PycharmProjects/HyperNetworks/BrainAgeDataset/metadata.csv")
    csv_subjects = list(csv.Subject)
    scans_subjects = os.listdir(data_dir)
    irelevant_subjects = set(scans_subjects) - set(csv_subjects)
    for subj in irelevant_subjects:
        subj_dir_path = os.path.join(data_dir, subj)
        shutil.rmtree(subj_dir_path)



a = 5