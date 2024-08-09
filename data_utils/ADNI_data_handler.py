import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import nibabel as nib
from .MetadataPreprocess import *
import pytorch_lightning as pl


class ADNIDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        transform_train = config.dataset_cfg.pop("transform_train")
        l2r_tform_train = config.dataset_cfg.pop("l2r_tform_train")
        transform_valid = config.dataset_cfg.pop("transform_valid")
        l2r_tform_valid = config.dataset_cfg.pop("l2r_tform_valid")
        stage = config.stage
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.train_ds = self.valid_ds = self.test_ds = None
        if stage == "train":
            self.train_ds = ADNI_Dataset(tr_val_tst="train", transform=transform_train,
                                         l2r_tform=l2r_tform_train, **config.dataset_cfg)
            self.valid_ds = ADNI_Dataset(tr_val_tst="valid", transform=transform_valid,
                                         l2r_tform=l2r_tform_valid, **config.dataset_cfg)

            if 1 > config.sample > 0:  # take a portion of the data (for debuggind the model)
                num_train_samples = int(len(self.train_ds) * config.sample)
                num_val_samples = int(len(self.valid_ds) * config.sample)
                self.train_ds = torch.utils.data.Subset(self.train_ds, np.arange(num_train_samples))
                self.valid_ds = torch.utils.data.Subset(self.valid_ds, np.arange(num_val_samples))


        elif stage == "test":
            self.test_ds = ADNI_Dataset(tr_val_tst="test", transform=transform_valid,
                                         l2r_tform=l2r_tform_valid, **config.dataset_cfg)


    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(dataset=self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)


class ADNI_Dataset(Dataset):
    def __init__(self, tr_val_tst, fold=0, features_set=5,
                 adni_dir='/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI',
                 transform=None, load2ram=False, rand_seed=2341, with_skull=False,
                 no_bias_field_correct=False, only_tabular=False, num_classes=3, split_seed=0,
                 l2r_tform=None, ADvsCN=False):
        self.tr_val_tst = tr_val_tst
        self.transform = tform_dict[transform]
        self.metadata = create_metadata_csv(features_set_idx=features_set, split_seed=split_seed, fold=fold)
        self.labels_dict = {
            2: {"CN": 0, 'AD': 1},
            3: {"CN": 0, 'MCI': 1, "AD": 2, 'EMCI': 1, "LMCI": 1},
            5: {"CN": 0, 'MCI': 1, "AD": 2, 'EMCI': 3, "LMCI": 4}
            }
        if num_classes == 2:  # keep AD and CN
            self.metadata = self.metadata[(self.metadata["Group"] == "CN") | (self.metadata["Group"] == "AD")]
            self.metadata.reset_index(inplace=True, drop=True)
        self.num_classes = num_classes
        self.labels_dict = self.labels_dict[num_classes]
        self.with_skull = with_skull
        self.no_bias_field_correct = no_bias_field_correct
        self.only_tabular = only_tabular

        self.num_tabular_features = len(self.metadata.columns) - 2  # the features excluding the Group and the Subject
        self.adni_dir = adni_dir
        assert fold in [0, 1, 2, 3]

        idxs_dict = self.get_folds_split(fold, split_seed)
        if tr_val_tst not in ['valid', 'train', 'test']:
            raise ValueError("tr_val_tst error: must be in ['valid', 'train', 'test']!!")
        self.metadata = self.metadata.loc[idxs_dict[tr_val_tst], :]  # tr_val_tst is valid, train or test

        self.metadata.reset_index(drop=True, inplace=True)

        self.data_in_ram = False
        self.imgs_ram_lst = []
        if load2ram:
            self.load_data2ram(l2r_tform)
            self.data_in_ram = True

    def get_folds_split(self, fold, split_seed=0):
        # ----------- split w.r.t the joint distribution of the label, sex & age -----------------
        df = self.metadata.copy()
        df = df.sample(frac=1, random_state=split_seed)
        df = df.replace(self.labels_dict)
        bins = 20
        df["AGE"] = pd.cut(df["AGE"], bins=bins, labels=[i for i in range(bins)])
        folds = [[], [], [], [], []]
        folds_idx = 0

        if "PTGENDER_Male" in df.columns:  # if sex is a feature
            for sex in df["PTGENDER_Male"].unique():
                for label in df["Group"].unique():
                    for age in np.sort(df["AGE"].unique()):
                        sub_df = df[(df["PTGENDER_Male"] == sex) & (df["Group"] == label) & (df["AGE"] == age)]
                        for idx in sub_df.index:
                            folds[folds_idx].append(idx)
                            folds_idx = (folds_idx + 1) % 5

        else:   # if there is AGE without sex
            for label in df["Group"].unique():
                for age in np.sort(df["AGE"].unique()):
                    sub_df = df[(df["Group"] == label) & (df["AGE"] == age)]
                    for idx in sub_df.index:
                        folds[folds_idx].append(idx)
                        folds_idx = (folds_idx + 1) % 5


        val_idxs = folds[fold]
        test_idxs = folds[4]

        train_idxs = list(np.where(~self.metadata.index.isin(list(val_idxs) + list(test_idxs)))[0])
        np.random.shuffle(train_idxs)
        idxs_dict = {'valid': val_idxs, 'train': train_idxs, 'test': test_idxs}
        return idxs_dict

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

    def load_data2ram(self, l2r_tform):
        assert l2r_tform is not None, "Used load to ram flag without specifying the relevant l2r_tform!"
        save_tform = self.transform  # save the regolar tform in this temp variable
        self.transform = tform_dict[l2r_tform]
        if self.tr_val_tst in ["valid", "test"]:
            loader = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=5)
            for batch in tqdm(loader, f'Loading {self.tr_val_tst} data to ram: '):
                self.imgs_ram_lst.append((batch[0][0].type(torch.float32), batch[1][0], batch[2][0]))

        if self.tr_val_tst == "train":
            loader = DataLoader(dataset=self, batch_size=1, shuffle=False, num_workers=20)
            for img, _, _ in tqdm(loader, f'Loading {self.tr_val_tst} data to ram: '):
                self.imgs_ram_lst.append(np.array(img[0, 0]))

        self.transform = save_tform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if self.data_in_ram:  # the data is a list alredy
            if self.tr_val_tst in ["valid", "test"]:
                img, features, label = self.imgs_ram_lst[index]
                return img.type(torch.float32), features, label
            if self.tr_val_tst == "train":
                img = self.imgs_ram_lst[index].copy()

        else:  # we need to load the data from the data dir
            subject = self.metadata.loc[index, "Subject"]
            if self.only_tabular:
                img = np.zeros((4, 4, 4, 4))
            else:
                img = self.load_image(subject)

        features = self.metadata.drop(['Subject', 'Group'], axis=1).loc[index]
        label = self.metadata.loc[index, "Group"]
        if self.only_tabular:
            return img, np.array(features, dtype=np.float32), self.labels_dict[label]

        img = img[None, ...]  # add channel dimention
        if not (self.transform is None):
            img = self.transform(img)

        return img, np.array(features, dtype=np.float32), self.labels_dict[label]


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

