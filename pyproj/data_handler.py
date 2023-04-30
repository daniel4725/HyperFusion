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


class ADNI_Dataset(Dataset):  # good rand seeds by order: [2341, 704, 795, 1557, 1977, 864, 2211, 2322, 13, 1991]
    def __init__(self, tr_val_tst, fold=0, metadata_path="metadata_by_features_sets/set-5.csv",
                 adni_dir='/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI',
                 transform=None, load2ram=False, rand_seed=2341, classes=("CN", "MCI", "AD"), with_skull=False,
                 no_bias_field_correct=False):
        self.tr_val_tst = tr_val_tst
        self.transform = transform
        self.metadata = pd.read_csv(metadata_path)
        self.with_skull = with_skull
        self.no_bias_field_correct = no_bias_field_correct

        self.num_tabular_features = len(self.metadata.columns) - 2  # the features excluding the Group and the Subject
        self.adni_dir = adni_dir

        # self.metadata = self.metadata[self.metadata["Subject"].isin(os.listdir(self.adni_dir))]  # take only the ones that are in the adni_dir
        # bad_images = [2028, 1664, 1562, 1500, 1365, 1364, 1363, 701, 449, 359, 277, 438, 450, 1317, 1465, 1782]
        # self.metadata = self.metadata.drop(index=bad_images)  # dropping unprocessed data
        # self.metadata.reset_index(drop=True, inplace=True)

        # to repeat the same data splits
        # default 'rand_seed' gives uniform distribution for both train and validation
        np.random.seed(rand_seed)
        indexes = [i for i in range(len(self.metadata))]
        np.random.shuffle(indexes)

        # split the data to the relevant fold
        assert fold in [0, 1, 2, 3, 4]
        fold_size = len(self.metadata) // 5  # fold size = 20%
        val_idxs = indexes[fold * fold_size:(fold + 1) * fold_size]
        test_idxs = indexes[4 * fold_size:]
        train_idxs = list(np.where(~self.metadata.index.isin(val_idxs + test_idxs))[0])
        idxs_dict = {'valid': val_idxs, 'train': train_idxs, 'test': test_idxs}

        if tr_val_tst not in ['valid', 'train', 'test']:
            raise ValueError("tr_val_tst error: must be in ['valid', 'train', 'test']!!")
        self.metadata = self.metadata.loc[idxs_dict[tr_val_tst], :]  # tr_val_tst is valid, train or test

        self.classes = classes
        if len(classes) == 2:  # if only CN - AD classes
            class_dict = {"CN": 0, "MCI": 1, "AD": 2}
            classes_idxs = (self.metadata.DX_bl == class_dict[classes[0]]) | (
                        self.metadata.DX_bl == class_dict[classes[1]])
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
        save_tform = self.transform  # save the regolar tform in this temp variable

        if self.tr_val_tst == "valid":
            self.transform = tform_dict["hippo_crop_2sides"]
            loader = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=5)
            for batch in tqdm(loader, f'Loading {self.tr_val_tst} data to ram: '):
                self.imgs_ram_lst.append((batch[0][0].type(torch.float32), batch[1][0], batch[2][0]))
            # for img, _, _ in tqdm(loader, f'Loading {self.tr_val_tst} data to ram: '):
            #     self.imgs_ram_lst.append(np.array(img[0, 0]))

        if self.tr_val_tst == "train":
            self.transform = tform_dict["hippo_crop_2sides_for_load_2_ram_func"]
            loader = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=20)
            # for batch in tqdm(loader, f'Loading {self.tr_val_tst} data to ram: '):
            #     self.imgs_ram_lst.append((batch[0][0], batch[1][0], batch[2][0]))
            for img, _, _ in tqdm(loader, f'Loading {self.tr_val_tst} data to ram: '):
                self.imgs_ram_lst.append(np.array(img[0, 0]))

        # for img, _, _ in tqdm(loader, f'Loading {self.tr_val_tst} data to ram: '):
        #     self.imgs_ram_lst.append(img[0, 0].type(torch.float32))

        if self.tr_val_tst == "test":
            raise ValueError("test l2r func not implemented yet!!")

        self.transform = save_tform

        # from utils import ThreadPoolEexecuter
        # def load_image2ram(subject, index):
        #     img = self.load_image(subject)
        #     img = tform_dict["hippo_crop_2sides_for_load_2_ram_func"](img[None])[0]
        #     self.imgs_ram_lst[index] = img
        #
        # img = self.load_image(os.listdir(self.adni_dir)[0])
        # thread_pool_executer = ThreadPoolEexecuter(num_workers=50)
        # self.imgs_ram_lst = [img] * len(self.metadata)
        # for idx in tqdm(range(len(self.metadata.Subject)), f'Loading {self.tr_val_tst} data to ram: '):
        #     subj = self.metadata.Subject[idx]
        #     thread_pool_executer.run_task(task=load_image2ram, args=[subj, idx])
        # thread_pool_executer.join_all()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if self.data_in_ram:  # the data is a list alredy
            if self.tr_val_tst == "valid":
                img, features, label = self.imgs_ram_lst[index]
                return img.type(torch.float32), features, label
                # img = self.imgs_ram_lst[index]
            if self.tr_val_tst == "train":
                img = self.imgs_ram_lst[index].copy()

        else:  # we need to load the data from the data dir
            subject = self.metadata.loc[index, "Subject"]
            img = self.load_image(subject)

        features = self.metadata.drop(['Subject', 'Group'], axis=1).loc[index]
        label = self.metadata.loc[index, "Group"]

        img = img[None, ...]  # add channel dimention
        if not (self.transform is None):
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

    torch.manual_seed(0)
    ADNI_dir = "/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI"
    # ADNI_dir = "/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/ADNI"
    # ADNI_dir = "/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/zipped_processed_data/ADNI"
    metadata_path = "metadata_by_features_sets/set-5.csv"
    num_workers = 3
    load2ram = True
    data_fold = 1
    tform = "hippo_crop_lNr_l2r_tst"  # center_crop  hippo_crop_lNr  hippo_crop_lNr_l2r


    loaders = get_dataloaders(batch_size=64, adni_dir=ADNI_dir, load2ram=load2ram,
                              metadata_path=metadata_path, fold=data_fold,
                              transform_train=tform_dict["hippo_crop_lNr_l2r"],
                              transform_valid=tform_dict["None"],
                              with_skull=False, no_bias_field_correct=True, num_workers=num_workers)
    train_loader0, valid_loader0 = loaders
    start = time.time()

    for batch in tqdm(train_loader0):
        a = 5
    for batch in tqdm(valid_loader0):
        a = 5

    print(time.time() - start)
    exit(0)

    valid_ds = ADNI_Dataset(tr_val_tst="valid", fold=0, metadata_path=metadata_path, adni_dir=ADNI_dir,
                            transform=tform_dict["None"], load2ram=True,
                            with_skull=False, no_bias_field_correct=True)

    valid_ds2 = ADNI_Dataset(tr_val_tst="valid", fold=0, metadata_path=metadata_path, adni_dir=ADNI_dir,
                             transform=tform_dict["hippo_crop_2sides"], load2ram=False,
                             with_skull=False, no_bias_field_correct=True)

    for i in tqdm(range(len(valid_ds))):
        d0 = valid_ds.__getitem__(i)
        d = valid_ds2.__getitem__(i)
        if not ((d0[0] == d[0]).all() and (d0[1] == d[1]).all() and (d0[2] == d[2])):
            print("valid error in ", i)



    loaders = get_dataloaders(batch_size=64, adni_dir=ADNI_dir, load2ram=load2ram,
                              metadata_path=metadata_path, fold=data_fold, transform_train=tform_dict[tform],
                              with_skull=False, no_bias_field_correct=True, num_workers=num_workers)

    # ----------------------------------------------------------------------------------
    # -------------------------   check if l2r works good    ---------------------------
    # ----------------------------------------------------------------------------------
    loaders = get_dataloaders(batch_size=64, adni_dir=ADNI_dir, load2ram=False,
                              metadata_path=metadata_path, fold=data_fold,
                              transform_train=tform_dict["hippo_crop_lNr_tst"],
                              transform_valid=tform_dict["hippo_crop_2sides"],
                              with_skull=False, no_bias_field_correct=True, num_workers=num_workers)
    train_loader0, valid_loader0 = loaders

    loaders = get_dataloaders(batch_size=64, adni_dir=ADNI_dir, load2ram=True,
                              metadata_path=metadata_path, fold=data_fold,
                              transform_train=tform_dict["hippo_crop_lNr_l2r_tst"],
                              transform_valid=tform_dict["normalize"],
                              with_skull=False, no_bias_field_correct=True, num_workers=num_workers)
    train_loader, valid_loader = loaders

    # for i in tqdm(range(len(train_loader.dataset))):
    #     d0 = train_loader0.dataset.__getitem__(i)
    #     d = train_loader.dataset.__getitem__(i)
    #     if not((d0[0] == d[0]).all() and (d0[1] == d[1]).all() and (d0[2] == d[2]).all()):
    #         print("train error in ", i)

    for i in tqdm(range(len(valid_loader.dataset))):
        d0 = valid_loader0.dataset.__getitem__(i)
        d = valid_loader.dataset.__getitem__(i)
        if not ((d0[0] == d[0]).all() and (d0[1] == d[1]).all() and (d0[2] == d[2])):
            print("valid error in ", i)

    # train:
    # error in 636
    # error in 643
    # error in 840
    #
    # valid:
    # error in 168

    a = 5
    # test_loader = get_test_loader(batch_size=5, metadata_path=metadata_path)

    # for i, (batch, tabular, y) in enumerate(train_loader):
    #     if i == 5:
    #         break
    #     for img in batch:
    #         imshow(img[0][100])

    # train_ds = train_loader.dataset
    # valid_ds = valid_loader.dataset
    # img, tabular, y = train_ds.__getitem__(0)

    # %%
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
    # %%
    # # run on y
    # img, tabular, y = train_ds.__getitem__(0)
    # y_start = 0
    # y_stop = img.shape[2] - 0
    # jumps = 1
    # for i in range(y_start,y_stop, jumps):
    #     plt.imshow(np.flip(img[0, :, i, :], 0), cmap='gray')
    #     plt.show()
    # %%
    # # run on x
    # img, tabular, y = train_ds.__getitem__(0)
    # x_start = 0
    # x_stop = img.shape[3] - 0
    # jumps = 4
    # for i in range(x_start,x_stop, jumps):
    #     plt.imshow(np.flip(img[0, :, :, i], 0), cmap='gray')
    #     plt.show()

    # %%
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

    # %%
    # find the best random seed for good target distribution:
    # good seeds (by order, better is right): [2341, 704, 795, 1557, 1977, 864, 2211, 2322, 13, 1991]
    # the best seed is 2341 with std=0.516
    # fold 0: train-[35. 48. 18.], valid-[36. 48. 16.]
    # fold 1: train-[35. 48. 17.], valid-[34. 48. 17.]
    # fold 2: train-[35. 48. 17.], valid-[34. 48. 18.]
    # fold 3: train-[35. 48. 17.], valid-[35. 48. 18.]
    # fold 4: train-[35. 48. 17.], valid-[34. 48. 17.]
    # valid std=[0.8   0.    0.748]

    # times = 2400
    # seeds_valid_std = []
    # for rand_seed in tqdm(range(times)):
    #     valid_histograms = []
    #     for fold in range(5):
    #         train_ds = ADNI_Dataset(tr_val_tst="train", fold=fold, adni_dir=ADNI_dir, metadata_path=metadata_path, rand_seed=rand_seed).metadata
    #         valid_ds = ADNI_Dataset(tr_val_tst="valid", fold=fold, adni_dir=ADNI_dir, metadata_path=metadata_path, rand_seed=rand_seed).metadata
    #
    #         train_target_hist = np.histogram(train_ds.Group, bins=3)[0]
    #         train_target_hist = np.round(train_target_hist/train_target_hist.sum(), 2) * 100
    #         valid_target_hist = np.histogram(valid_ds.Group, bins=3)[0]
    #         valid_target_hist = np.round(valid_target_hist/valid_target_hist.sum(), 2) * 100
    #         valid_histograms.append(valid_target_hist)
    #         #print(f"fold {fold}: train-{train_target_hist}, valid-{valid_target_hist}")
    #     #print(f"valid std={np.round(np.std(valid_histograms, axis=0), 3)}")
    #     seeds_valid_std.append(np.round(np.std(valid_histograms, axis=0), 3))
    # weighted_seeds_valid_std = np.mean(seeds_valid_std, axis=1)
    # best_seed = weighted_seeds_valid_std.argmin()
    # print(f"the best seed is {best_seed} with std={weighted_seeds_valid_std.min():.3f}")
    # k = 10
    # res = sorted(range(len(weighted_seeds_valid_std)), key=lambda sub: weighted_seeds_valid_std[sub])[:k]
    # print(f"the best {k} seeds by order are: {res}")
    #
    # valid_histograms = []
    # for fold in range(5):
    #     train_ds = ADNI_Dataset(tr_val_tst="train", fold=fold, adni_dir=ADNI_dir, metadata_path=metadata_path, rand_seed=best_seed).metadata
    #     valid_ds = ADNI_Dataset(tr_val_tst="valid", fold=fold, adni_dir=ADNI_dir, metadata_path=metadata_path, rand_seed=best_seed).metadata
    #
    #     train_target_hist = np.histogram(train_ds.Group, bins=3)[0]
    #     train_target_hist = np.round(train_target_hist/train_target_hist.sum(), 2) * 100
    #     valid_target_hist = np.histogram(valid_ds.Group, bins=3)[0]
    #     valid_target_hist = np.round(valid_target_hist/valid_target_hist.sum(), 2) * 100
    #     valid_histograms.append(valid_target_hist)
    #     print(f"fold {fold}: train-{train_target_hist}, valid-{valid_target_hist}")
    # print(f"valid std={np.round(np.std(valid_histograms, axis=0), 3)}")

    # ----------------------------------------------------------------------------------
    # ------------------------- check if the images are good---------------------------
    # ----------------------------------------------------------------------------------
    # nan_inf_lst_1 = []
    # nan_inf_lst_2 = []
    # nan_inf_lst_mask = []
    # csv = pd.read_csv(metadata_path)
    # csv = csv[csv["Subject"].isin(os.listdir(ADNI_dir))]  # take only the ones that are in the adni_dir
    # csv = csv.reset_index()
    # for i in tqdm(range(len(csv))):
    # # for i in tqdm(range(450, len(csv))):
    #     subject = csv.loc[i, "Subject"]
    #     img1_path = os.path.join(ADNI_dir, subject, "brain_scan_simple.nii.gz")
    #     img2_path = os.path.join(ADNI_dir, subject, "brain_scan.nii.gz")
    #     mask_path = os.path.join(ADNI_dir, subject, "brain_mask.nii.gz")
    #
    #     img1 = nib.load(img1_path).get_fdata()
    #     img2 = nib.load(img2_path).get_fdata()
    #     mask = nib.load(mask_path).get_fdata()
    #
    #     # if np.isnan(img1).any() or np.isinf(img1).any():
    #     #     nan_inf_lst_1.append(i)
    #     #     print(f"img1: index is inf or nan - {i}")
    #     # if np.isnan(img2).any() or np.isinf(img2).any():
    #     #     nan_inf_lst_2.append(i)
    #     #     print(f"img2: index is inf or nan - {i}")
    #     if not (set(np.unique(mask)) == set([0., 1.])):
    #         nan_inf_lst_mask.append(i)
    #         print(f"mask: index is inf or nan or mask is not 0,1 - {i}")
    #
    #     if (img1 < 0).any() or (img1 > 100000).any():
    #         print(f"img1: not in range. index - {i}")
    #     if (img2 < 0).any() or (img2 > 100000).any():
    #         print(f"img2: not in range. index - {i}")

    # mask: index is inf or nan or mask is not 0, 1 - 438
    # img1: index is inf or nan - 450
    # mask: index is inf or nan or mask is not 0, 1 - 1317
    # img1: index is inf or nan - 1465
    # img2: index is inf or nan - 1782

    # img1: not in range.index - 277
    # img2: not in range.index - 359
    # img2: not in range.index - 449
    # img1: not in range.index - 701
    # img1: not in range.index - 1363
    # img1: not in range.index - 1364
    # img1: not in range.index - 1365
    # img2: not in range.index - 1500
    # img1: not in range.index - 1562
    # img2: not in range.index - 1664
    # img2: not in range.index - 2028

    # i = 438
    csv = pd.read_csv(metadata_path)
    csv = csv[csv["Subject"].isin(os.listdir(ADNI_dir))]  # take only the ones that are in the adni_dir
    csv = csv.reset_index()


    def inspect_idx(i):
        subject = csv.loc[i, "Subject"]
        img1_path = os.path.join(ADNI_dir, subject, "brain_scan_simple.nii.gz")
        img2_path = os.path.join(ADNI_dir, subject, "brain_scan.nii.gz")
        mask_path = os.path.join(ADNI_dir, subject, "brain_mask.nii.gz")
        img1 = nib.load(img1_path).get_fdata()
        img2 = nib.load(img2_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        print(f"img1 max: {img1.max()}")
        print(f"img1 min: {img1.min()}")
        plt.hist(img1.flatten())
        plt.title("img1")
        plt.show()
        print(f"img2 max: {img2.max()}")
        print(f"img2 min: {img2.min()}")
        plt.hist(img2.flatten())
        plt.title("img2")
        plt.show()


    print("-------- end data handler --------")