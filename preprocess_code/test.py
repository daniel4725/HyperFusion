import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import nibabel as nib
def imshow(img):
    plt.imshow(img, cmap="gray")
    plt.show()


# adni_dir = os.path.join(os.getcwd(), "ADNI")
# subj_list = ["34", "76", "94", "s121", "s122", "s124"]
#
# precessed_img_format = "/media/data2/tempDaniel/working_samples/_subject_id_{}/numpySave_simple/numPyArray_{}.npy"
# raw_img_format = "/media/data2/tempDaniel/working_samples/{}/raw.nii.gz"
# for i, subject in enumerate(subj_list):
#
#     raw_path = raw_img_format.format(subject)
#     # raw_path = "/media/data2/tempDaniel/robex_tests/strip_094_S_1330.nii.gz"
#     processed_path = precessed_img_format.format(subject, subject[-4:])
#
#     raw_img = nib.load(raw_path).get_fdata()
#     raw_img = (255 * (raw_img - np.min(raw_img)) / np.ptp(raw_img)).astype("uint8")
#     raw_slices = list(range(0, raw_img.shape[2], 5))
#
#     processed_img = np.load(processed_path)
#     processed_img = (255 * (processed_img - np.min(processed_img)) / np.ptp(processed_img)).astype("uint8")
#     processed_slices = list(range(0, processed_img.shape[2], 5))
#
#     print(f"{i}: {subject}, raw_shape={raw_img.shape}, processed_shape={processed_img.shape}")
#
#     for s in raw_slices:
#         img_slice = raw_img[:, :, s]
#         cv2.imshow("raw img", cv2.resize(img_slice, (0, 0), fx=3, fy=3))
#
#         if cv2.waitKey(170) != -1:
#             print("Stopped!")
#             cv2.waitKey(0)
#
#     for s in processed_slices:
#         img_slice = processed_img[:, :, s]
#         cv2.imshow("processed img", cv2.resize(img_slice, (0, 0), fx=3, fy=3))
#
#         if cv2.waitKey(170) != -1:
#             print("Stopped!")
#             cv2.waitKey(0)




adni_dir = os.path.join(os.getcwd(), "ADNI")
subj_list = os.listdir(adni_dir)

precessed_img_format = "/media/data2/tempDaniel/processed/Preprocess_Pipeline_V2/_subject_id_{}/numpySave_simple/numPyArray_{}.npy"
# precessed_img_format = "/media/data2/tempDaniel/processed/Preprocess_Pipeline_V2/_subject_id_{}/brain_mask.npy"
raw_img_format = adni_dir + "/{}/raw.nii"
for i, subject in enumerate(subj_list):

    raw_path = raw_img_format.format(subject)
    processed_path = precessed_img_format.format(subject, subject[-4:])

    # raw_img = nib.load(raw_path).get_fdata()
    # raw_img = (255 * (raw_img - np.min(raw_img)) / np.ptp(raw_img)).astype("uint8")
    # raw_slices = list(range(0, raw_img.shape[2], 5))

    processed_img = np.load(processed_path)
    processed_img = (255 * (processed_img - np.min(processed_img)) / np.ptp(processed_img)).astype("uint8")
    # processed_img = processed_img[16:-16,16:-16, :]
    processed_slices = list(range(0, processed_img.shape[0], 5))
    processed_slices = list(range(processed_img.shape[0]))

    # print(f"{i}: {subject}, raw_shape={raw_img.shape}, processed_shape={processed_img.shape}")
    print(f"{i}: {subject}, raw_shape=, processed_shape={processed_img.shape}")

    # for s in raw_slices:
    #     img_slice = raw_img[:, :, s, 0]
    #     cv2.imshow("raw img", cv2.resize(img_slice, (0, 0), fx=3, fy=3))
    #
    #     if cv2.waitKey(170) != -1:
    #         print("Stopped!")
    #         cv2.waitKey(0)

    for s in processed_slices:
        img_slice = processed_img[s, :, :]
        cv2.imshow("processed img", cv2.resize(img_slice, (0, 0), fx=3, fy=3))

        if cv2.waitKey(170) != -1:
            print("Stopped!")
            cv2.waitKey(0)

"""
_subject_id_114_S_0374
_subject_id_126_S_0708
_subject_id_126_S_0709
_subject_id_126_S_0784
_subject_id_126_S_0865
_subject_id_126_S_0891
_subject_id_126_S_1077
"""



