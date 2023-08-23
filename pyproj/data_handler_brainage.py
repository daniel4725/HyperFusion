# %%
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2
import nibabel as nib
from tformNaugment import tform_dict
from sklearn.model_selection import StratifiedKFold
from scipy.stats import entropy
from MetadataPreprocess import *


def imshow(img):
    plt.imshow(img, cmap="gray")
    plt.show()


def scanshow(img):
    normalized_img = (255 * (img - np.min(img)) / np.ptp(img)).astype("uint8")
    for img_slice in normalized_img:
        cv2.imshow("scan show", cv2.resize(img_slice, (0, 0), fx=3, fy=3))
        if cv2.waitKey(70) != -1:
            print("Stopped!")
            cv2.waitKey(0)


class BrainAgeDataset(Dataset):
    def __init__(self, tr_val_tst, fold=0, features_set=5,
                 adni_dir='/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI',
                 transform=None, load2ram=False, rand_seed=2341, with_skull=False,
                 no_bias_field_correct=False, only_tabular=False, num_classes=3):
        self.tr_val_tst = tr_val_tst
        self.transform = transform
        self.dataset_path = "/home/duenias/PycharmProjects/HyperNetworks/BrainAgeDataset"
        self.metadata = pd.read_csv(os.path.join(self.dataset_path, "metadata.csv"))
        self.metadata = pd.get_dummies(self.metadata, dummy_na=False, columns=["Gender"])

        self.num_tabular_features = 2  # the features excluding the Group and the Subject
        assert fold in [0, 1, 2, 3]
        if tr_val_tst not in ['valid', 'train', 'test']:
            raise ValueError("tr_val_tst error: must be in ['valid', 'train', 'test']!!")

        idxs_dict = self.get_folds_split(fold)
        self.metadata = self.metadata.loc[idxs_dict[tr_val_tst], :]  # tr_val_tst is valid, train or test

        self.metadata.reset_index(drop=True, inplace=True)

    def get_folds_split(self, fold, rand_seed=0):
        np.random.seed(0)
        skf = StratifiedKFold(n_splits=5, random_state=rand_seed, shuffle=True)  # 1060 is good seed for joint distribution
        X = self.metadata.drop(['Subject', 'Age'], axis=1)

        y = self.metadata["Age"].round().astype(int)

        list_of_splits = list(skf.split(X, y))
        _, val_idxs = list_of_splits[fold]
        _, test_idxs = list_of_splits[4]

        train_idxs = list(np.where(~self.metadata.index.isin(list(val_idxs) + list(test_idxs)))[0])
        np.random.shuffle(train_idxs)
        idxs_dict = {'valid': val_idxs, 'train': train_idxs, 'test': test_idxs}
        return idxs_dict

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        subject = self.metadata.loc[index, "Subject"]
        img = np.load(os.path.join(self.dataset_path, "brain_scans", subject, "brain_scan.npy"))

        features = self.metadata.drop(['Subject', 'Age'], axis=1).loc[index]
        label = self.metadata.loc[index, "Age"]

        img = img[None, ...]  # add channel dimention
        if not (self.transform is None):
            img = self.transform(img)

        return img, np.array(features, dtype=np.float32), label


def get_dataloaders(batch_size, features_set=5,
                    adni_dir='/usr/local/faststorage/adni_class_pred_2x2x2_v1', fold=0, num_workers=0,
                    transform_train=None, transform_valid=None, load2ram=False, sample=1,
                    with_skull=False, no_bias_field_correct=True, only_tabular=False, num_classes=3):
    """ creates the train and validation data sets and creates their data loaders"""
    train_ds = BrainAgeDataset(tr_val_tst="train", fold=fold, features_set=features_set, adni_dir=adni_dir,
                            transform=transform_train, load2ram=load2ram, only_tabular=only_tabular,
                            with_skull=with_skull, no_bias_field_correct=no_bias_field_correct, num_classes=num_classes)
    valid_ds = BrainAgeDataset(tr_val_tst="valid", fold=fold, features_set=features_set, adni_dir=adni_dir,
                            transform=transform_valid, load2ram=load2ram, only_tabular=only_tabular,
                            with_skull=with_skull, no_bias_field_correct=no_bias_field_correct, num_classes=num_classes)

    if sample < 1 and sample > 0:  # take a portion of the data (for debuggind the model)
        num_train_samples = int(len(train_ds) * sample)
        num_val_samples = int(len(valid_ds) * sample)
        train_ds = torch.utils.data.Subset(train_ds, np.arange(num_train_samples))
        valid_ds = torch.utils.data.Subset(valid_ds, np.arange(num_val_samples))

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader


def get_test_loader(batch_size, features_set=5,
                    adni_dir='/usr/local/faststorage/adni_class_pred_2x2x2_v1', fold=0, num_workers=0,
                    transform=None, load2ram=False, num_classes=3,
                    with_skull=False, no_bias_field_correct=False, only_tabular=False):
    """ creates the test data set and creates its data loader"""
    test_ds = BrainAgeDataset(tr_val_tst="test", fold=fold, features_set=features_set, adni_dir=adni_dir,
                            transform=transform, load2ram=load2ram, only_tabular=only_tabular,
                            with_skull=with_skull, no_bias_field_correct=no_bias_field_correct, num_classes=num_classes)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader


if __name__ == "__main__":
    from tformNaugment import tform_dict

    test_loader = get_test_loader(batch_size=32)
    train_loader, val_loader = get_dataloaders(batch_size=32, fold=0)

    batch = next(iter(train_loader))

    print("-------- end data handler --------")