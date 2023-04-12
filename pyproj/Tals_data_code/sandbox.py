import shutil
import os
import cv2
import numpy as np

def scanshow(img):
    normalized_img = (255 * (img - np.min(img)) / np.ptp(img)).astype("uint8")
    for img_slice in normalized_img:
        cv2.imshow("scan show", cv2.resize(img_slice, (0, 0), fx=3, fy=3))
        if cv2.waitKey(70) != -1:
            print("Stopped!")
            cv2.waitKey(0)

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
no_good_subj_lst = ["128_S_2045", "127_S_5058", "024_S_6846", "941_S_1203", "098_S_4201", "136_S_0194", "027_S_0256"]
base_path = "/home/duenias/PycharmProjects/HyperNetworks/processed"
for subj in no_good_subj_lst:
    img_path = os.path.join(base_path, subj, "brainscan.npy")
    img = np.load(img_path)
    scanshow(img)