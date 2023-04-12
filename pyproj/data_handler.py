#%%
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import time
import lz4.frame
import cv2
import nibabel as nib

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

class ADNI_Dataset(Dataset):
    def __init__(self, tr_val_tst, fold=0, metadata_path="metadata_by_features_sets/set-5.csv", adni_dir='/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI',
                 transform=None, load2ram=False, rand_seed=2341, classes=("CN", "MCI", "AD"), with_skull=False, no_bias_field_correct=False):
        self.tr_val_tst = tr_val_tst
        self.transform = transform
        self.metadata = pd.read_csv(metadata_path)
        self.with_skull = with_skull
        self.no_bias_field_correct = no_bias_field_correct

        self.num_tabular_features = len(self.metadata.columns) - 2  # the features excluding the Group and the Subject
        self.adni_dir = adni_dir
        self.metadata = self.metadata[self.metadata["Subject"].isin(os.listdir(self.adni_dir))]  # take only the ones that are in the adni_dir
        self.metadata.reset_index(drop=True, inplace=True)

        # to repeat the same data splits
        # default 'rand_seed' gives uniform distribution for both train and validation
        np.random.seed(rand_seed)  
        indexes = [i for i in range(len(self.metadata))]
        np.random.shuffle(indexes)

        # split the data to the relevant fold
        assert fold in [0, 1, 2, 3, 4]
        fold_size = len(self.metadata)//5  # fold size = 20% 
        val_idxs = indexes[fold * fold_size:(fold + 1) * fold_size]
        test_idxs = indexes[4 * fold_size:]
        train_idxs = list(np.where(~self.metadata.index.isin(val_idxs + test_idxs))[0])
        idxs_dict = {'valid': val_idxs, 'train': train_idxs, 'test': test_idxs}

        if tr_val_tst not in ['valid', 'train', 'test']:
                    raise ValueError("tr_val_tst error: must be in ['valid', 'train', 'test']!!")
        self.metadata = self.metadata.loc[idxs_dict[tr_val_tst], :]  # tr_val_tst is valid, train or test

        self.classes = classes
        if len(classes) == 2:  # if only CN - AD classes
            class_dict = {"CN":0, "MCI":1, "AD":2}
            classes_idxs = (self.metadata.DX_bl == class_dict[classes[0]]) | (self.metadata.DX_bl == class_dict[classes[1]])
            self.metadata = self.metadata[classes_idxs]
            self.metadata.DX_bl.iloc[self.metadata.DX_bl == class_dict[classes[0]]] = 0
            self.metadata.DX_bl.iloc[self.metadata.DX_bl == class_dict[classes[1]]] = 1

        self.metadata.reset_index(drop=True, inplace=True)

        self.data_in_ram = False
        self.imgs_ram_lst = []
        if load2ram:
            self.load_data2ram()
            self.data_in_ram = True

    def load_image(self, subject):
        if self.no_bias_field_correct:
            img_path = os.path.join(self.adni_dir, subject, "brain_scan_simple.nii.gz")
        else:
            img_path = os.path.join(self.adni_dir, subject, "brain_scan.nii.gz")
        img = nib.load(img_path).get_fdata()
        if not self.with_skull:
            mask_path = os.path.join(self.adni_dir, subject, "brain_mask.nii.gz")
            img = img * nib.load(mask_path).get_fdata()  # apply the brain mask
        return img

    def load_image_npy(self, subject):
        if self.no_bias_field_correct:
            img_path = os.path.join(self.adni_dir, subject, "brain_scan_simple.npy")
        else:
            img_path = os.path.join(self.adni_dir, subject, "brain_scan.npy")
        img = np.load(img_path)
        if not self.with_skull:
            mask_path = os.path.join(self.adni_dir, subject, "brain_mask.npy")
            img = img * np.load(mask_path)  # apply the brain mask
        return img

    def load_data2ram(self):
        # loads the data to the ram (make a list of all the loaded data)
        for i in tqdm(range(len(self.metadata)), f'Loading {self.tr_val_tst} data to ram: '):
            subject = self.metadata.loc[i, "Subject"]
            img = self.load_image(subject)
            self.imgs_ram_lst.append(img)
    

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if self.data_in_ram:   # the data is a list alredy and ir is normalized
            img = self.imgs_ram_lst[index].copy()
        else:     # we need to load the data from the data dir
            subject = self.metadata.loc[index, "Subject"]
            img = self.load_image(subject)
            
        features = self.metadata.drop(['Subject', 'Group'], axis=1).loc[index]
        label = self.metadata.loc[index, "Group"]
        
        img = img[None, ...]  # add channel dimention
        if not(self.transform is None):
            img = self.transform(img)

        return img, np.array(features, dtype=np.float32), label

        
def get_dataloaders(batch_size, metadata_path="metadata_by_features_sets/set-1.csv",
                    adni_dir='/usr/local/faststorage/adni_class_pred_2x2x2_v1', fold=0, num_workers=0,
                    transform_train=None, transform_valid=None, load2ram=False, sample=1, classes=("CN", "MCI", "AD"),
                    with_skull=False, no_bias_field_correct=False):
    """ creates the train and validation data sets and creates their data loaders"""
    train_ds = ADNI_Dataset(tr_val_tst="train", fold=fold, metadata_path=metadata_path, adni_dir=adni_dir,
                           transform=transform_train, load2ram=load2ram, classes=classes,
                            with_skull=with_skull, no_bias_field_correct=no_bias_field_correct)
    valid_ds = ADNI_Dataset(tr_val_tst="valid", fold=fold, metadata_path=metadata_path, adni_dir=adni_dir,
                           transform=transform_valid, load2ram=load2ram, classes=classes,
                            with_skull=with_skull, no_bias_field_correct=no_bias_field_correct)

    if sample < 1 and sample > 0:  # take a portion of the data (for debuggind the model)
        num_train_samples = int(len(train_ds) * sample)
        num_val_samples = int(len(valid_ds) * sample)
        train_ds = torch.utils.data.Subset(train_ds, np.arange(num_train_samples))
        valid_ds = torch.utils.data.Subset(valid_ds, np.arange(num_val_samples))

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader

def get_test_loader(batch_size, metadata_path="metadata_features_set_1.csv",
                    num_workers=0, load2ram=False, transform=None, classes=("CN", "MCI", "AD"),
                    with_skull=False, no_bias_field_correct=False):
    """ creates the test data set and creates its data loader"""
    test_ds = ADNI_Dataset(tr_val_tst="test", metadata_path=metadata_path,
                           transform=transform, load2ram=load2ram, classes=classes,
                            with_skull=with_skull, no_bias_field_correct=no_bias_field_correct)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader

if __name__ == "__main__":
    from tformNaugment import tform_dict
    ADNI_dir = "/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI"
    # ADNI_dir = "/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/ADNI"
    # ADNI_dir = "/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/zipped_processed_data/ADNI"
    metadata_path = "metadata_by_features_sets/set-5.csv"
    data_fold = 0

    loaders = get_dataloaders(batch_size=5, adni_dir=ADNI_dir, load2ram=False,
                             metadata_path=metadata_path, fold=data_fold, transform_train=tform_dict["hippo_crop_lNr"],
                              with_skull=False, no_bias_field_correct=True)
    train_loader, valid_loader = loaders
    img, tabular, y = train_loader.dataset.__getitem__(1)
    # test_loader = get_test_loader(batch_size=5, metadata_path=metadata_path)

    # for i, (batch, tabular, y) in enumerate(train_loader):
    #     if i == 5:
    #         break
    #     for img in batch:
    #         imshow(img[0][100])

    # train_ds = train_loader.dataset
    # valid_ds = valid_loader.dataset
    # img, tabular, y = train_ds.__getitem__(0)

#%%
    # # run on z
    # for k in [10, 30, 70, 120, 200]:
    #     img, tabular, y = train_ds.__getitem__(k)
    #     print(f"----------- y = {y} ---------------")
    #     z_start = 25
    #     z_stop = z_start + 64
    #     y_start = 55
    #     y_stop = y_start + 96
    #     x_start = 88
    #     x_stop = x_start + 64
    #     img = img[:, z_start: z_stop, y_start: y_stop, x_start: x_stop]
    #     jumps = 2
    #     for i in range(0,img.shape[1], jumps):
    #         plt.imshow(img[0, i], cmap='gray')
    #         plt.show()
#%%
    # # run on y
    # img, tabular, y = train_ds.__getitem__(0)
    # y_start = 0
    # y_stop = img.shape[2] - 0
    # jumps = 1
    # for i in range(y_start,y_stop, jumps):
    #     plt.imshow(np.flip(img[0, :, i, :], 0), cmap='gray')
    #     plt.show()
#%%
    # # run on x
    # img, tabular, y = train_ds.__getitem__(0)
    # x_start = 0
    # x_stop = img.shape[3] - 0
    # jumps = 4
    # for i in range(x_start,x_stop, jumps):
    #     plt.imshow(np.flip(img[0, :, :, i], 0), cmap='gray')
    #     plt.show()


#%%
    # # create histograms of the tabular data
    #
    # ds = train_ds.append(valid_ds, ignore_index=True)
    # ds.DX_bl[ds.DX_bl == 0] = "CN"
    # ds.DX_bl[ds.DX_bl == 1] = "MCI"
    # ds.DX_bl[ds.DX_bl == 2] = "AD"
    # ds["PTGENDER"] = ds.PTGENDER_Female
    # ds["PTGENDER"][ds.PTGENDER_Female == 1] = "Female"
    # ds["PTGENDER"][ds.PTGENDER_Female == 0] = "Male"
    #
    #
    # genetic = ["APOE4"] # genetic Risk factors
    # demographics = ["PTGENDER", "PTEDUCAT", "AGE"]
    # Cognitive = ["CDRSB", "ADAS13", "ADAS11", "MMSE", "RAVLT_immediate"]
    #
    # for name in genetic + demographics + Cognitive:
    #     plt.figure()
    #     plt.hist(ds[name][ds.DX_bl == "CN"], alpha=0.5, bins=12, range=(0, 1))
    #     plt.hist(ds[name][ds.DX_bl == "MCI"], alpha=0.5, bins=12, range=(0, 1))
    #     plt.hist(ds[name][ds.DX_bl == "AD"], alpha=0.5, bins=12, range=(0, 1))
    #     plt.legend(["CN", "MCI", "AD"])
    #     plt.title(f"{name}")
    #
    #
    #
    #




#%%
    # find the best random seed for good target distribution:
    # good seeds (by order, better is right): 0, 13, 2341
    # the best seed is 2341 with std=0.516
    # fold 0: train-[35. 48. 18.], valid-[36. 48. 16.]
    # fold 1: train-[35. 48. 17.], valid-[34. 48. 17.]
    # fold 2: train-[35. 48. 17.], valid-[34. 48. 18.]
    # fold 3: train-[35. 48. 17.], valid-[35. 48. 18.]
    # fold 4: train-[35. 48. 17.], valid-[34. 48. 17.]
    # valid std=[0.8   0.    0.748]

    times = 3000
    seeds_valid_std = []
    for rand_seed in tqdm(range(times)):
        valid_histograms = []
        for fold in range(5):
            train_ds = ADNI_Dataset(tr_val_tst="train", fold=fold, adni_dir=ADNI_dir, metadata_path=metadata_path, rand_seed=rand_seed).metadata
            valid_ds = ADNI_Dataset(tr_val_tst="valid", fold=fold, adni_dir=ADNI_dir, metadata_path=metadata_path, rand_seed=rand_seed).metadata

            train_target_hist = np.histogram(train_ds.Group, bins=3)[0]
            train_target_hist = np.round(train_target_hist/train_target_hist.sum(), 2) * 100
            valid_target_hist = np.histogram(valid_ds.Group, bins=3)[0]
            valid_target_hist = np.round(valid_target_hist/valid_target_hist.sum(), 2) * 100
            valid_histograms.append(valid_target_hist)
            #print(f"fold {fold}: train-{train_target_hist}, valid-{valid_target_hist}")
        #print(f"valid std={np.round(np.std(valid_histograms, axis=0), 3)}")
        seeds_valid_std.append(np.round(np.std(valid_histograms, axis=0), 3))
    weighted_seeds_valid_std = np.mean(seeds_valid_std, axis=1)
    best_seed = weighted_seeds_valid_std.argmin()
    print(f"the best seed is {best_seed} with std={weighted_seeds_valid_std.min():.3f}")
    
    valid_histograms = []
    for fold in range(5):
        train_ds = ADNI_Dataset(tr_val_tst="train", fold=fold, adni_dir=ADNI_dir, metadata_path=metadata_path, rand_seed=best_seed).metadata
        valid_ds = ADNI_Dataset(tr_val_tst="valid", fold=fold, adni_dir=ADNI_dir, metadata_path=metadata_path, rand_seed=best_seed).metadata

        train_target_hist = np.histogram(train_ds.Group, bins=3)[0]
        train_target_hist = np.round(train_target_hist/train_target_hist.sum(), 2) * 100
        valid_target_hist = np.histogram(valid_ds.Group, bins=3)[0]
        valid_target_hist = np.round(valid_target_hist/valid_target_hist.sum(), 2) * 100
        valid_histograms.append(valid_target_hist)
        print(f"fold {fold}: train-{train_target_hist}, valid-{valid_target_hist}")
    print(f"valid std={np.round(np.std(valid_histograms, axis=0), 3)}")
    
    print("-------- end data handler --------")    