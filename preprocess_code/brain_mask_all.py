def RobexImage(inputImg, out_mask, out_stripped):
    # Skull stripping with Robex. Please choose the right bash_command depends on computer
    import os
    # chooses relveant Bash_command depends on computer
    bash_commands = {
        'L': 'bash /media/galia/Data1/Softwares/ROBEX/runROBEX.sh ' + inputImg + ' ' + out_stripped + ' ' + out_mask,
        'M': 'bash /media/data1/software/ROBEXBrainExtraction-master/ROBEX/runROBEX.sh ' + inputImg + ' ' + out_stripped + ' ' + out_mask,
        'R': 'bash /media/galia-lab/Data1/software/ROBEX/runROBEX.sh ' + inputImg + ' ' + out_stripped + ' ' + out_mask}
    bashCommand = bash_commands.get('M')
    # Runs bash command
    import subprocess
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output, error)
    return out_stripped, out_mask

if __name__ == "__main__":
    import os
    import numpy as np
    import nibabel as nib
    from tqdm import tqdm
    processed_dir = "/media/data2/tempDaniel/processed_done"
    robex_mask_dir = "/media/data2/tempDaniel/processed_done_robex_masks2"
    os.makedirs(robex_mask_dir, exist_ok=True)

    for i, subj in enumerate(os.listdir(processed_dir)[:5]):
        print(f"starting - [{i + 1}/{len(os.listdir(processed_dir))}]: {subj}")
        subj_dir = os.path.join(robex_mask_dir, subj)
        if os.path.isfile(os.path.join(subj_dir, "brain_mask.npy")):  # if there is a mask alredy
            continue
        os.makedirs(subj_dir, exist_ok=True)
        img = np.load(os.path.join(processed_dir, subj, "brain_scan_simple.npy")).T
        nifti_image = nib.Nifti1Image(img, affine=np.eye(4))
        nifti_image_path = os.path.join(subj_dir, "nifti_image.nii.gz")
        nib.save(nifti_image, nifti_image_path)
        mask_path = os.path.join(subj_dir, "mask.nii.gz")
        stripped_path = os.path.join(subj_dir, "stripped.nii.gz")
        RobexImage(nifti_image_path, mask_path, stripped_path)

        mask = nib.load(mask_path).get_fdata().T
        np.save(os.path.join(subj_dir, "brain_mask.npy"), mask)

        os.remove(nifti_image_path)
        os.remove(stripped_path)
        os.remove(os.path.join(subj_dir, "brain_mask.npy"))
        # os.remove(mask_path)





