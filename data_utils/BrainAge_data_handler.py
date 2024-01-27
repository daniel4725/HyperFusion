import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import shutil

def make_video(img3d):
    img3d = np.pad(img3d, ((19, 19), (4, 4), (0, 0)))
    x, y, z = img3d.shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter("video" + '.mp4', fourcc, 10, (x, y))
    for i in range(z):
        img_slice = img3d[:, :, z - i - 1]
        img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)

        img_slice *= 255
        cv2.imshow("s", np.array(img_slice).astype('uint8'))
        video.write(np.array(img_slice).astype('uint8'))
        cv2.waitKey(30)

    video.release()

def copy_data_to_server(dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    mri_ds = MRIDataset(data_in_storage_server=True)
    for i in tqdm(range(len(mri_ds))):
        img_path, subject = mri_ds.get_path_and_subject(i)
        dest_path = os.path.join(dest_dir, subject + ".npy")
        shutil.copyfile(img_path, dest_path)


def get_mri_dataloaders(batch_size, num_workers=0, gender=None, data_in_storage_server=False, partial_data=False,
                        ages=None, transform=None):
    if not (gender in [None, "M", "F"]):
        raise ValueError('gender must be None, "M" or "F" !!')
    valid_ds = MRIDataset(gender=gender, data_type="valid", data_in_storage_server=data_in_storage_server,
                          partial_data=partial_data, ages=ages, transform=transform)
    test_ds = MRIDataset(gender=gender, data_type="test", data_in_storage_server=data_in_storage_server,
                         partial_data=partial_data, ages=ages, transform=transform)
    train_ds = MRIDataset(gender=gender, data_type="train", data_in_storage_server=data_in_storage_server,
                          partial_data=partial_data, ages=ages, transform=transform)

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


def get_data_size(list_of_data_loaders):
    size = 0
    for loader in list_of_data_loaders:
        size += len(loader)
    return size


class MRIDataset(Dataset):
    def __init__(self, gender=None, data_type=None, metadata_path=None, data_dir=None,
                 transform=None, data_in_storage_server=False, partial_data=False, ages=None):

        if metadata_path is None:
            if data_type is None:
                metadata_path = "metadata/metadata_age_prediction.csv"
            else:
                metadata_path = f"metadata/metadata_age_prediction_{data_type}.csv"
        if data_dir is None:
            if data_in_storage_server:
                data_dir = "/media/rrtammyfs/labDatabase/BrainAge/Healthy"
            else:
                data_dir = os.path.join(os.getcwd(), "data")
        metadata = pd.read_csv(metadata_path)
        if partial_data:
            data_end = int(len(metadata) * partial_data)
            metadata = metadata.loc[:data_end]
        if gender is None:
            self.metadata = metadata
        else:
            self.metadata = metadata.loc[metadata["Gender"] == gender].reset_index(drop=True)
        if ages is not None:
            ages_idx = (self.metadata["Age"] >= ages[0]) & (self.metadata["Age"] < ages[1])
            self.metadata = self.metadata.loc[ages_idx].reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.data_in_storage_server = data_in_storage_server

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        subject = self.metadata.loc[index, "Subject"]
        if self.data_in_storage_server:
            img_path = os.path.join(self.data_dir, subject, "numpySave", f"{subject}.npy")
        else:
            img_path = os.path.join(self.data_dir, f"{subject}.npy")

        img = np.load(img_path)
        gender = self.metadata.loc[index, "Gender"]
        age = self.metadata.loc[index, "Age"]

        if self.transform is not None:
            img = self.transform(img)

        return img[None, ...], gender, age  # add channel axis to the image

    def get_path_and_subject(self, index):
        subject = self.metadata.loc[index, "Subject"]
        if self.data_in_storage_server:
            img_path = os.path.join(self.data_dir, subject, "numpySave", f"{subject}.npy")
        else:
            img_path = os.path.join(self.data_dir, f"{subject}.npy")
        return img_path, subject


def create_MRI_metadata(csv_path):
    csv = pd.read_csv(csv_path)
    relevant_gender = (csv["Gender"] == 'F') | (csv["Gender"] == 'M')
    relevant_lines = relevant_gender & csv["Age"].notnull()

    metadata = csv.loc[relevant_lines, ["Subject", "Gender", "Age", "ProjName"]]

    metadata = metadata.sample(frac=1).reset_index(drop=True)
    metadata.to_csv("metadata_age_prediction.csv", index=False)

    train = metadata.sample(frac=0.8, random_state=0)  # 80% train
    rest_of_data = metadata.loc[~metadata.index.isin(train.index)]

    valid = rest_of_data.sample(frac=0.5, random_state=0)  # 50% of 20% = 10% validation
    test = rest_of_data.loc[~rest_of_data.index.isin(valid.index)]  # 10% test

    train.to_csv("metadata_age_prediction_train.csv", index=False)
    valid.to_csv("metadata_age_prediction_valid.csv", index=False)
    test.to_csv("metadata_age_prediction_test.csv", index=False)


if __name__ == "__main__":
    data_path = "/home/duenias/PycharmProjects/HyperNetworks/BrainAgeDataset/data"
    copy_data_to_server(data_path)
    # copy_data_to_server("data")

    # data_loaders = get_mri_dataloaders(batch_size=1, num_workers=0, partial_data=False)
    # train_loader, valid_loader, test_loader = data_loaders
    #
    # batch, gender, age = train_loader.dataset.__getitem__(0)
    # img = batch[0]
    # img = img - img.min()
    # img = img / img.max()
    # make_video(img)


    # base_csv_path = "/media/rrtammyfs/labDatabase/BrainAge/Healthy_subjects_divided_pipe_v2.csv"
    # # create_MRI_metadata(base_csv_path)
    #
    # data_loaders = get_mri_dataloaders(batch_size=16, num_workers=0, gender="M", partial_data=False)
    # train_M_loader, valid_M_loader, test_M_loader = data_loaders
    #
    # data_loaders = get_mri_dataloaders(batch_size=16, num_workers=0, gender="F", partial_data=False)
    # train_F_loader, valid_F_loader, test_F_loader = data_loaders
    #
    # data_loaders = get_mri_dataloaders(batch_size=16, num_workers=0, partial_data=False)
    # train_loader, valid_loader, test_loader = data_loaders
    #
    # data_loaders = get_mri_dataloaders(batch_size=16, num_workers=0, partial_data=0.5)
    # train_loader_p, valid_loader_p, test_loader_p = data_loaders
    #
    # print(2)
    # for n in range(11):
    #     whole_loader = DataLoader(dataset=mri_ds, batch_size=16, shuffle=True, num_workers=n)
    #     start = time.time()
    #     t_before = start
    #     times = []
    #     for i, (batch, gender, age) in enumerate(whole_loader):
    #         t = time.time() - t_before
    #         t_before = time.time()
    #         times.append(t)
    #         if i == 10:
    #             break
    #     print(f"{n}: total = {np.mean(times)}")

        # for img in batch:
        #     img = img - img.min()
        #     img = img / img.max()
        #     for i in range(img.shape[2]):
        #         cv2.imshow("s", np.array(img[:, :, i] * 255).astype('uint8'))
        #         cv2.waitKey(10)

    # shapes = []
    # for i in tqdm(range(len(mri_ds))):
    #     img, gender, age = mri_ds.__getitem__(i)
    #     shapes.append(img.shape)

    # with open('shapes', 'wb') as file:
    #     pickle.dump(shapes, file)
    #
    # print(set(shapes), len(shapes))
    #
    # with open('shapes', 'rb') as file:
    #     s = pickle.load(file)


