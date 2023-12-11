import os
import shutil
from tqdm import tqdm

def keep_numpy_simple(subj_dir):
    # Functions that saves transformations and shapes from pre chosen steps.
    # The function also deletes heavy memory files to decrease pipeline output memory usage
    import os
    import pickle
    import nibabel as nib
    import shutil
    output_dirs = os.listdir(subj_dir)
    output_dirs.remove("numpySave_simple")
    for dir in output_dirs:
        shutil.rmtree(os.path.join(subj_dir, dir))

#  ---------------------------------------------------------------------------
# # move the done parts to ADNI_done
# slice_save_dir = os.path.join(os.getcwd(), "processed/Slices/Complexed/xSlice")
# done_lst = [name[:-4] for name in os.listdir(slice_save_dir)]
# adni_dir = os.path.join(os.getcwd(), "ADNI")
# dst = os.path.join(os.getcwd(), "ADNI_done")
# for subj in done_lst:
#     src = os.path.join(adni_dir, subj)
#     shutil.move(src, dst)

# # move all the full processed to processed_done
# processed_dir = os.path.join(os.getcwd(), "processed/Preprocess_Pipeline_V2")
# dst = os.path.join(os.getcwd(), "processed_done")
# for subj in done_lst:
#     src = os.path.join(processed_dir, f"_subject_id_{subj}")
#     shutil.move(src, dst)

# # check if all processed have numpy simple
# processed_done_dir = os.path.join(os.getcwd(), "processed_done")
# flag = "All Ok!"
# for subj in os.listdir(processed_done_dir):
#     subj_dir = os.path.join(processed_done_dir, subj)
#     if not(("numpySave" in os.listdir(subj_dir)) and ("numpySave_simple" in os.listdir(subj_dir))):
#         print(f"{subj} missing numpy")
#         flag = "Not Ok!"
#     if not("brain_mask.npy" in os.listdir(subj_dir)):
#         print(f"{subj} missing numpy")
#         flag = "Not Ok!"
# print(flag)

#  ---------------------------------------------------------------------------
# deleting all the folders except numpy simple

processed_dir = os.path.join(os.getcwd(), "processed", "Preprocess_Pipeline_V2")
subjects = [dir for dir in os.listdir(processed_dir) if dir[:2] == "_s"]
# for subj in subjects:
#     dir = os.path.join(processed_dir, subj)
#     keep_numpy_simple(dir)


#  ---------------------------------------------------------------------------
#  moves all the scans to the main subject's folder and changes the name to raw.nii

adni_dir = os.path.join(os.getcwd(), "ADNI")
subjects = os.listdir(adni_dir)
# scans_paths = []
# for subject in subjects:
#     subj_dir = os.path.join(adni_dir, subject)
#     for path, dirs, files in os.walk(subj_dir):
#         if files != []:
#             src_path = os.path.join(path, files[0])
#             dst_path = os.path.join(subj_dir, "raw.nii")
#             if src_path != dst_path:
#                 shutil.move(src_path, dst_path)
#


# --------------------------------------------------------------------------
# moves the numpy simple image to the patient dir and deletes the rest at the end
base_dir = os.path.join(os.getcwd(), "processed_done")

subjects_dirs = [os.path.join(base_dir, directory) for directory in os.listdir(base_dir)]
for subj_dir in tqdm(subjects_dirs):
    subj_id = subj_dir[-10:]
    npy_simple_path = os.path.join(subj_dir, f"numpySave_simple/numPyArray_{subj_id[-4:]}.npy")
    np_path = os.path.join(subj_dir, f"numpySave/numPyArray_{subj_id[-4:]}.npy")
    mask_path = os.path.join(subj_dir, f"brain_mask.npy")
    new_subj_dir = os.path.join(base_dir, subj_id)
    os.makedirs(new_subj_dir)
    shutil.move(np_path, os.path.join(new_subj_dir, "brain_scan.npy"))
    shutil.move(npy_simple_path, os.path.join(new_subj_dir, "brain_scan_simple.npy"))
    shutil.move(mask_path, os.path.join(new_subj_dir, "brain_mask.npy"))
    # deletes the rest
    shutil.rmtree(subj_dir)
