import shutil
import os
import cv2
import numpy as np
from tqdm import tqdm
import nibabel as nib
import threading
import time
import pandas as pd

class ThreadPoolEexecuter:
    def __init__(self, num_workers):
        import queue
        self.threads_queue = queue.Queue(num_workers)

    def run_task(self, task, args):
        if self.threads_queue.full():
            t = self.threads_queue.get()
            t.join()
        t = threading.Thread(target=task, args=args)
        t.start()
        self.threads_queue.put(t)



base_dir = "/media/rrtammyfs/labDatabase/ADNI/ADNI_2023"
# base_dir = "/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023"
adni_dir = base_dir + "/ADNI"
metadata_path = base_dir + "/my_adnimerege.csv"
metadata = pd.read_csv(metadata_path)

# metadata = metadata[metadata["Subject"].isin(os.listdir(adni_dir))]  # take only the ones that are in the adni_dir
# bad_images = [2028, 1664, 1562, 1500, 1365, 1364, 1363, 701, 449, 359, 277, 438, 450, 1317, 1465, 1782]
# metadata = metadata.drop(index=bad_images)  # dropping unprocessed data
# metadata.reset_index(drop=True, inplace=True)
# metadata.to_csv(metadata_path, index=False)

a = 0
for subj in os.listdir(adni_dir):
    if subj not in list(metadata["Subject"]):
        subj_path = adni_dir + '/' + subj
        shutil.rmtree(subj_path)
        a += 1
        print(a)

# # copy the .nii.gz files to DGX server
# src = "/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/zipped_processed_data/ADNI"
# dst = "/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI"
# shutil.copytree(src, dst)




# thread_pool_executer = ThreadPoolEexecuter(num_workers=20)
# # iterate over all the data
# adni_dir = "/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/ADNI"
# subjects = os.listdir(adni_dir)
# for subj in tqdm(subjects):
#     path = os.path.join(adni_dir, subj)

    # # threading copy data to DGX server
    # src = os.path.join("/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/zipped_processed_data/ADNI", subj)
    # dst = os.path.join("/home/duenias/PycharmProjects/HyperNetworks/ADNI_2023/ADNI", subj)
    # thread_pool_executer.run_task(task=shutil.copytree, args=[src, dst])

    # # check that all the subjects have ['brain_scan_simple.npy', 'brain_mask.npy', 'brain_scan.npy']
    # files_list = os.listdir(path)
    # if files_list != ['brain_scan_simple.npy', 'brain_mask.npy', 'brain_scan.npy']:
    #     print(subj)


    # # save all numpy files as nii.gz for smaller capacity
    # dst = os.path.join("/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/zipped_processed_data/ADNI", subj)
    # os.makedirs(dst, exist_ok=True)
    #
    # brain_scan_simple = np.load(os.path.join(path, 'brain_scan_simple.npy'))
    # brain_scan_simple = nib.Nifti1Image(brain_scan_simple, affine=np.eye(4))
    # nib.save(brain_scan_simple, os.path.join(dst, 'brain_scan_simple.nii.gz'))
    #
    # brain_scan = np.load(os.path.join(path, 'brain_scan.npy'))
    # brain_scan = nib.Nifti1Image(brain_scan, affine=np.eye(4))
    # nib.save(brain_scan, os.path.join(dst, 'brain_scan.nii.gz'))
    #
    # brain_mask = np.load(os.path.join("/media/rrtammyfs/labDatabase/ADNI/ADNI_2023/processed_done_robex_masks", subj, 'brain_mask.npy'))
    # brain_mask = nib.Nifti1Image(brain_mask, affine=np.eye(4))
    # nib.save(brain_mask, os.path.join(dst, 'brain_mask.nii.gz'))


    # # shows the scans
    # img_s = np.load(os.path.join(path, "brain_scan_simple.npy"))
    # img = np.load(os.path.join(path, "brain_scan.npy"))
    # mask = np.load(os.path.join(path, "brain_mask.npy"))
    # scanshow(img_s)
    # scanshow(img)
    # scanshow(mask * img_s)


#
# base_dir = os.path.join(os.getcwd(), "processed")
#
# inner_base_dir = os.path.join(base_dir, "Preprocess_Pipeline_V2")
# subjects_dirs = [os.path.join(inner_base_dir, directory) for directory in os.listdir(inner_base_dir) if directory[:2] == "_s"]
# for subj_dir in subjects_dirs:
#     subj_id = subj_dir[-10:]
#     img_path = os.path.join(subj_dir, f"numpySave_simple/numPyArray_{subj_id[-4:]}.npy")
#     new_subj_dir = os.path.join(base_dir, subj_id)
#     os.makedirs(new_subj_dir)
#     shutil.move(img_path, os.path.join(new_subj_dir, "brainscan.npy"))
# shutil.rmtree(inner_base_dir)

# not so good ones:
# no_good_subj_lst = ["128_S_2045", "127_S_5058", "024_S_6846", "941_S_1203", "098_S_4201", "136_S_0194", "027_S_0256"]
# base_path = "/home/duenias/PycharmProjects/HyperNetworks/processed"
# for subj in no_good_subj_lst:
#     img_path = os.path.join(base_path, subj, "brainscan.npy")
#     img = np.load(img_path)
#     scanshow(img)

