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

def load_image(path: str):
    if path.endswith('.npy.lz4'):
        with lz4.frame.open(path, 'rb') as f:
            return np.load(f)
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise Exception("File extension not supported!")

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
    def __init__(self, tr_val_tst, fold=0, metadata_path="metadata_features_set_1.csv", adni_dir='/usr/local/faststorage/adni_class_pred_1x1x1_v1',
                 transform=None, load2ram=False, rand_seed=2729, classes=["CN", "MCI", "AD"]):
        self.tr_val_tst = tr_val_tst
        self.transform = transform
        self.metadata = pd.read_csv(metadata_path)
        # ----------------------------------------
        # TODO delete this part
        # adni_new_dir = "/home/duenias/PycharmProjects/HyperNetworks/adni_class_pred_1x1x1_v1/ADNI"
        # for dir in os.listdir(adni_new_dir):
        #     files = os.listdir(os.path.join(adni_new_dir, dir))
        #     print(files)
        #     if ("brainmask.npy" in files) and ("t1p.npy" in files):
        #         continue
        #     else:
        #         print("!!!! not good enough!!!")
        # print("FINISHED ALL")
        #
        # new_base = "/home/duenias/PycharmProjects/HyperNetworks/adni_class_pred_1x1x1_v1"
        # for img_path_end, mask_path_end in tqdm(zip(self.metadata["IMAGE_PATH"], self.metadata["MASK_PATH"])):
        #     img_path = os.path.join(adni_dir, img_path_end)
        #     mask_path = os.path.join(adni_dir, mask_path_end)
        #     new_dir = os.path.join(new_base, os.path.split(img_path_end)[0])
        #     os.makedirs(new_dir, exist_ok=True)
        #
        #     img = np.load(img_path)
        #     mask = np.load(mask_path)
        #
        #     np.save(os.path.join(new_base, img_path_end), img)
        #     np.save(os.path.join(new_base, mask_path_end), mask)

        #     new_img_path
        #     new_mask_path
        #
        #     with lz4.frame.open(img_path, 'rb') as f:
        #         img = np.load(f)
        #         np.save(img_path[:-4], img)
        #
        #     with lz4.frame.open(mask_path, 'rb') as f:
        #         mask = np.load(f)
        #         np.save(mask_path[:-4], mask)
        #
        #     print(img_path)
        #     print(mask_path)

        # m_path = metadata_path.replace("1", "8")
        # self.metadata = pd.read_csv(m_path)
        # self.metadata["IMAGE_PATH"] = self.metadata["IMAGE_PATH"].str[:-4]
        # self.metadata["MASK_PATH"] = self.metadata["MASK_PATH"].str[:-4]
        # self.metadata.to_csv(m_path, index=False)

        # ----------------------------------------

        self.num_tabular_features = len(self.metadata.columns) - 3  # the features excluding the label and the paths
        self.adni_dir = adni_dir

        # to repeat the same data splits
        # default 'rand_seed' gives uniform distribution for both train and validation
        np.random.seed(rand_seed)  
        indexes = [i for i in range(len(self.metadata))]
        np.random.shuffle(indexes)

        # split the data to the relevant fold
        assert fold in [0, 1, 2, 3]
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

    def load_data2ram(self):
        # loads the data to the ram (make a list of all the loaded data)
        for i in tqdm(range(len(self.metadata)), f'Loading {self.tr_val_tst} data to ram: '):
            img = load_image(os.path.join(self.adni_dir, self.metadata.loc[i, "IMAGE_PATH"]))
            mask = load_image(os.path.join(self.adni_dir, self.metadata.loc[i, "MASK_PATH"]))
            img = img * mask # apply the brain mask
            self.imgs_ram_lst.append(img)
    
    @staticmethod
    def strip_skull_and_normalize(img, mask):
        img = img * mask
        return (img - img.mean())/img.std()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if self.data_in_ram:   # the data is a list alredy and ir is normalized
            img = self.imgs_ram_lst[index].copy()
        else:     # we need to load the data from the data dir
            img = load_image(os.path.join(self.adni_dir, self.metadata.loc[index, "IMAGE_PATH"]))
            mask = load_image(os.path.join(self.adni_dir, self.metadata.loc[index, "MASK_PATH"]))
            img = img * mask  # apply the brain mask
            
        features = self.metadata.drop(['DX_bl', 'IMAGE_PATH', "MASK_PATH"], axis=1).loc[index]
        label = self.metadata.loc[index, "DX_bl"]
        
        img = img[None, ...]  # add channel dimention
        if not(self.transform is None):
            img = self.transform(img)

        return img, np.array(features, dtype=np.float32), label

        
def get_dataloaders(batch_size, metadata_path="metadata_by_features_sets/set-1.csv",
                    adni_dir='/usr/local/faststorage/adni_class_pred_2x2x2_v1', fold=0, num_workers=0,
                    transform_train=None, transform_valid=None, load2ram=False, sample=1, classes=["CN", "MCI", "AD"]):
    """ creates the train and validation data sets and creates their data loaders"""
    train_ds = ADNI_Dataset(tr_val_tst="train", fold=fold, metadata_path=metadata_path, adni_dir=adni_dir,
                           transform=transform_train, load2ram=load2ram, classes=classes)
    valid_ds = ADNI_Dataset(tr_val_tst="valid", fold=fold, metadata_path=metadata_path, adni_dir=adni_dir,
                           transform=transform_valid, load2ram=load2ram, classes=classes)

    if sample < 1 and sample > 0:  # take a portion of the data (for debuggind the model)
        num_train_samples = int(len(train_ds) * sample)
        num_val_samples = int(len(valid_ds) * sample)
        train_ds = torch.utils.data.Subset(train_ds, np.arange(num_train_samples))
        valid_ds = torch.utils.data.Subset(valid_ds, np.arange(num_val_samples))

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader

def get_test_loader(batch_size, metadata_path="metadata_features_set_1.csv",
                    num_workers=0, load2ram=False, transform=None, classes=["CN", "MCI", "AD"]):
    """ creates the test data set and creates its data loader"""
    test_ds = ADNI_Dataset(tr_val_tst="test", metadata_path=metadata_path,
                           transform=transform, load2ram=load2ram, classes=classes)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader

if __name__ == "__main__":
    # ADNI_dir = "/usr/local/faststorage/adni_class_pred_2x2x2_v1"
    ADNI_dir = "/usr/local/faststorage/adni_class_pred_1x1x1_v1"
    metadata_path = "metadata_by_features_sets/set-1.csv"
    data_fold = 2

    adni_csv = pd.read_csv(os.path.join(ADNI_dir, "ADNI.csv"))
    adni_csv = adni_csv[adni_csv.IMAGEUID != 34537]  # 34537 this one is not in the directories
    adni_csv = adni_csv.reset_index(drop=True)

    loaders = get_dataloaders(batch_size=5, adni_dir=ADNI_dir,
                             metadata_path=metadata_path,
                             fold=data_fold)
    train_loader, valid_loader = loaders
    test_loader = get_test_loader(batch_size=5, metadata_path=metadata_path)

    train_ds = train_loader.dataset
    valid_ds = valid_loader.dataset
    img, tabular, y = train_ds.__getitem__(0)

#%%
    # run on z
    for k in [10, 30, 70, 120, 200]:
        img, tabular, y = train_ds.__getitem__(k)
        print(f"----------- y = {y} ---------------")
        z_start = 25
        z_stop = z_start + 64
        y_start = 55
        y_stop = y_start + 96
        x_start = 88
        x_stop = x_start + 64
        img = img[:, z_start: z_stop, y_start: y_stop, x_start: x_stop]
        jumps = 2
        for i in range(0,img.shape[1], jumps):
            plt.imshow(img[0, i], cmap='gray')
            plt.show()
#%%
    # run on y
    img, tabular, y = train_ds.__getitem__(0)
    y_start = 0
    y_stop = img.shape[2] - 0
    jumps = 1
    for i in range(y_start,y_stop, jumps):
        plt.imshow(np.flip(img[0, :, i, :], 0), cmap='gray')
        plt.show()
#%%
    # run on x
    img, tabular, y = train_ds.__getitem__(0)
    x_start = 0
    x_stop = img.shape[3] - 0
    jumps = 4
    for i in range(x_start,x_stop, jumps):
        plt.imshow(np.flip(img[0, :, :, i], 0), cmap='gray')
        plt.show()


#%%
    # create histograms of the tabular data

    ds = train_ds.append(valid_ds, ignore_index=True)
    ds.DX_bl[ds.DX_bl == 0] = "CN" 
    ds.DX_bl[ds.DX_bl == 1] = "MCI" 
    ds.DX_bl[ds.DX_bl == 2] = "AD" 
    ds["PTGENDER"] = ds.PTGENDER_Female
    ds["PTGENDER"][ds.PTGENDER_Female == 1] = "Female"
    ds["PTGENDER"][ds.PTGENDER_Female == 0] = "Male"
     

    genetic = ["APOE4"] # genetic Risk factors
    demographics = ["PTGENDER", "PTEDUCAT", "AGE"]
    Cognitive = ["CDRSB", "ADAS13", "ADAS11", "MMSE", "RAVLT_immediate"]

    for name in genetic + demographics + Cognitive:
        plt.figure()
        plt.hist(ds[name][ds.DX_bl == "CN"], alpha=0.5, bins=12, range=(0, 1))
        plt.hist(ds[name][ds.DX_bl == "MCI"], alpha=0.5, bins=12, range=(0, 1))
        plt.hist(ds[name][ds.DX_bl == "AD"], alpha=0.5, bins=12, range=(0, 1))
        plt.legend(["CN", "MCI", "AD"])
        plt.title(f"{name}")








#%%
    # find the best random seed for good target distribution:
    # good seeds (by order, better is right): 8, 30, 658, 2729 
    # seed 2729 distribution:
        # fold 0: train-[26. 58. 17.], valid-[26. 57. 17.]
        # fold 0: train-[26. 57. 17.], valid-[25. 59. 16.]
        # fold 0: train-[26. 58. 16.], valid-[25. 56. 19.]
        # fold 0: train-[26. 57. 17.], valid-[26. 59. 15.]
        # fold 0: train-[25. 58. 17.], valid-[26. 58. 16.]

    times = 3000
    seeds_valid_std = []
    for rand_seed in tqdm(range(times)):
        valid_histograms = []
        for fold in range(5):
            train_ds = ADNI_Dataset(tr_val_tst="train", fold=fold, metadata_path=metadata_path, rand_seed=rand_seed).metadata
            valid_ds = ADNI_Dataset(tr_val_tst="valid", fold=fold, metadata_path=metadata_path, rand_seed=rand_seed).metadata

            train_target_hist = np.histogram(train_ds.DX_bl, bins=3)[0]
            train_target_hist = np.round(train_target_hist/train_target_hist.sum(), 2) * 100
            valid_target_hist = np.histogram(valid_ds.DX_bl, bins=3)[0]
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
        train_ds = ADNI_Dataset(tr_val_tst="train", fold=fold, metadata_path=metadata_path, rand_seed=best_seed).metadata
        valid_ds = ADNI_Dataset(tr_val_tst="valid", fold=fold, metadata_path=metadata_path, rand_seed=best_seed).metadata

        train_target_hist = np.histogram(train_ds.DX_bl, bins=3)[0]
        train_target_hist = np.round(train_target_hist/train_target_hist.sum(), 2) * 100
        valid_target_hist = np.histogram(valid_ds.DX_bl, bins=3)[0]
        valid_target_hist = np.round(valid_target_hist/valid_target_hist.sum(), 2) * 100
        valid_histograms.append(valid_target_hist)
        print(f"fold {fold}: train-{train_target_hist}, valid-{valid_target_hist}")
    print(f"valid std={np.round(np.std(valid_histograms, axis=0), 3)}")
    
    print("-------- end data handler --------")    