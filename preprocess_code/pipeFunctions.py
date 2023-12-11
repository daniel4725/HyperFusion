def rescaleImage(inputImg):
    # resamples inputImage with a new affine transformation according to a chosen constant
    import nibabel as nib
    import os
    import numpy as np
    from nilearn.image import resample_img
    # extracts subject's name, and defines function's name
    sub_id = inputImg.split('/')[6][12:]
    funcName = 'rescaledImage_'
    # defines rescale constant
    const = 1.75
    # loads image and its' affine
    img = nib.load(inputImg)
    imgAffine = np.array(img.affine)


    # # creates new affine trasnformation for rescale
    # rescale_mat = np.eye(4) * const
    # rescale_mat[3, 3] = 1
    # imgAffine = np.dot(imgAffine, rescale_mat)
    # # resmaples inputImage to the new affine trasnformation and saves it
    # nImg = resample_img(img, imgAffine)
    # out_path = os.path.join(os.getcwd(), funcName + str(sub_id) + '.nii.gz')
    # nib.save(nImg, out_path)

    def pad2shape(arr, shape=(256, 256, 256)):
        shapes = [((i_ - i) // 2, (i_ - i) // 2 + (i_ - i) % 2) for i_, i in zip(shape, arr.shape)]
        # print(f"---------------------------\n\n\n {arr.shape}\n{shapes} \n\n\n\n ---------------------------")
        return np.pad(arr, shapes, mode='constant')
    img = pad2shape(img.get_fdata())
    img = nib.Nifti1Image(img, affine=imgAffine)
    out_path = os.path.join(os.getcwd(), funcName + str(sub_id) + '.nii.gz')
    nib.save(img, out_path)
    return out_path

def center(image_array):
    # inner function used in centerImage in Brain_Age_Pipeline_V1
    import numpy as np
    x, y, z = np.nonzero(image_array)
    x_s, x_e, x_len = x.min(), x.max(), image_array.shape[0]
    y_s, y_e, y_len = y.min(), y.max(), image_array.shape[1]
    z_s, z_e, z_len = z.min(), z.max(), image_array.shape[2]
    vector = (x_s - (x_len - x_e), y_s - (y_len - y_e), z_s - (z_len - z_e))
    for i in range(3):
        if vector[i] != 0:
            image_array = np.roll(image_array, (vector[i] // 2) * (-1), i)
    return image_array


def centerImage(inputImg):
    # centers image, used in Brain_Age_Pipeline_V1
    import nibabel as nib
    import numpy as np
    import pipeFunctions
    funcName = 'centeredImage_'
    sub_id = inputImg.split('/')[6][12:]
    img = nib.load(inputImg)
    image_array = np.squeeze(img.get_data())
    image_array = pipeFunctions.center(image_array)
    out_path = pipeFunctions.saveImageAsNifty(image_array, img.affine, sub_id, funcName)
    return out_path


# def cropImage(inputImg):
#     # crops image in the middle to desired path, used in Brain_Age_Pipeline_V1
#     import nibabel as nib
#     import numpy as np
#     import pipeFunctions
#     # extracts subject's name, and defines function's name
#     sub_id = inputImg.split('/')[6][12:]
#     funcName = 'croppedImage_'
#     # loads img and its' array
#     img = nib.load(inputImg)
#     img_array = img.get_data()
#     # checks shape .vs. desired shape for cropping purposes
#     shape = img_array.shape
#     desierd_shape = [90, 120, 99]
#     # integer list for differences
#     diff = [0, 0, 0]
#     # boolean list for differences
#     diff_check = [0, 0, 0]
#     cropped_array = np.zeros(desierd_shape)
#     # checks differences in each axis
#     for i in range(3):
#         if shape[i] > desierd_shape[i]:
#             diff[i] = (shape[i] - desierd_shape[i]) // 2
#         else:
#             diff[i] = (desierd_shape[i] - shape[i]) // 2
#             diff_check[i] = 1
#         if abs(shape[i] - desierd_shape[i]) % 2 != 0:
#             diff[i] = diff[i] + 1
#     # crops according to 1 of 8 possible diff states
#     if diff_check == [0, 0, 0]:
#         cropped_array = img_array[diff[0]:(desierd_shape[0] + diff[0]), diff[1]:(desierd_shape[1] + diff[1]),
#                         diff[2]:(desierd_shape[2] + diff[2])]
#     else:
#         if diff_check == [0, 0, 1]:
#             cropped_array[:, :, diff[2]:(shape[2] + diff[2])] = img_array[diff[0]:(desierd_shape[0] + diff[0]),
#                                                                 diff[1]:(desierd_shape[1] + diff[1]), :]
#         else:
#             if diff_check == [0, 1, 0]:
#                 cropped_array[:, diff[1]:(shape[1] + diff[1]), :] = img_array[diff[0]:(desierd_shape[0] + diff[0]), :,
#                                                                     diff[2]:(desierd_shape[2] + diff[2])]
#             else:
#                 if diff_check == [1, 0, 0]:
#                     cropped_array[diff[0]:(shape[0] + diff[0]), :, :] = img_array[:,
#                                                                         diff[1]:(desierd_shape[1] + diff[1]),
#                                                                         diff[2]:(desierd_shape[2] + diff[2])]
#                 else:
#                     if diff_check == [0, 1, 1]:
#                         cropped_array[:, diff[1]:(shape[1] + diff[1]), diff[2]:(shape[2] + diff[2])] = \
#                             img_array[diff[0]:(desierd_shape[0] + diff[0]), :, :]
#                     else:
#                         if diff_check == [1, 0, 1]:
#                             cropped_array[diff[0]:(shape[0] + diff[0]), :, diff[2]:(shape[2] + diff[2])] = \
#                                 img_array[:, diff[1]:(desierd_shape[1] + diff[1]), :]
#                         else:
#                             if diff_check == [1, 1, 0]:
#                                 cropped_array[diff[0]:(shape[0] + diff[0]), diff[1]:(shape[1] + diff[1]), :] \
#                                     = img_array[:, :, diff[2]:(desierd_shape[2] + diff[2])]
#                             else:
#                                 if diff_check == [1, 1, 1]:
#                                     cropped_array[diff[0]:(shape[0] + diff[0]),
#                                     diff[1]:(shape[1] + diff[1]), diff[2]:(shape[2] + diff[2])] = \
#                                         img_array[:, :, :]
#     # centers and save the cropped array
#     cropped_array = pipeFunctions.center(cropped_array)
#     cropped_path = pipeFunctions.saveImageAsNifty(cropped_array, img.affine, sub_id, funcName)
#     return cropped_path


def imageIntesity(inputImg, maskFile):
    # normalises inputImage intensity
    import intensity_normalization as fcm
    import os
    import nibabel as nib
    # extracts subject's name, and defines function's name
    sub_id = inputImg.split('/')[6][12:]
    funcName = 'intensedImage_'
    # loads image and mask file
    vol = nib.load(inputImg)
    mask = nib.load(maskFile)
    # apply masking and normalization for the inputImage
    wm_mask = fcm.find_wm_mask(vol, mask)
    normalized = fcm.fcm_normalize(vol, wm_mask)
    # saves and returns normalized image
    normalizedImage = os.path.join(os.getcwd(), funcName + str(sub_id) + '.nii.gz')
    nib.save(normalized, normalizedImage)
    return normalizedImage


def RobexImage(inputImg):
    # Skull stripping with Robex. Please choose the right bash_command depends on computer
    import os
    # extracts subject's name
    sub_id = inputImg.split('/')[6][12:]
    # defines mask and strupped file location
    out_mask = os.path.join(os.getcwd(), 'mask_file_' + str(sub_id) + '.nii.gz')
    out_stripped = os.path.join(os.getcwd(), 'stripped_file_' + str(sub_id) + '.nii.gz')
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


def saveAsNumpy(inputImg):
    # saves numpy array of the complexed pipeline output at the end of the pipeline
    import nibabel as nib
    import numpy as np
    import os
    funcName = 'numPyArray_'
    sub_id = inputImg.split('/')[-1].split('_')[-1].split('.')[0]
    input_array = nib.load(inputImg).get_data().T
    out_path = os.path.join(os.getcwd(), funcName + str(sub_id) + '.npy')
    np.save(out_path, input_array)
    return out_path


def maximum_filter(inputImg):
    # extends inputImage with chosen parameter with 'maximum_filter' function
    from scipy import ndimage
    import nibabel as nib
    import pipeFunctions
    import os
    import numpy as np
    # extracts subject's name, and defines function's name
    sub_id = inputImg.split('/')[6][12:]
    funcName = 'filteredImage_'
    # loads inputImage and its' numpy array
    img = nib.load(inputImg)
    input_array = img.get_data()
    # applies maximum_filter and saves the new image
    filtered = ndimage.maximum_filter(input_array, size=22)
    filtered_path = pipeFunctions.saveImageAsNifty(filtered, img.affine, sub_id, funcName)

    # saves the mask without the skull
    x, y, z = np.nonzero(input_array)
    x_s, x_e = x.min(), x.max()
    y_s, y_e = y.min(), y.max()
    z_s, z_e = z.min(), z.max()
    brain_mask = input_array[x_s: x_e, y_s: y_e, z_s: z_e]

    des_shape = [170, 186, 170]
    shapes = [((i_ - i) // 2, (i_ - i) // 2 + (i_ - i) % 2) for i_, i in zip(des_shape, brain_mask.shape)]
    brain_mask = np.pad(brain_mask, shapes, mode='constant').T

    masks_dir = os.path.dirname(os.getcwd())
    out_path = masks_dir + '/brain_mask.npy'
    np.save(out_path, brain_mask)

    return filtered_path


# def save_regular_mask(img, sub_id):
#     # save the brain mask without the max filter
#     import numpy as np
#     import os
#     from nipype.interfaces.fsl import ExtractROI

#     masks_dir = f'/media/data2/tempDaniel/processed/Preprocess_Pipeline_V2/_subject_id_{sub_id}/brain_masks'
#     masks_dir = os.getcwd() + '/brain_masks'
#     os.mkdir(masks_dir)

#     # pad the image
#     input_array = img.get_fdata()
#     shape=(256, 256, 256)
#     shapes = [((i_ - i) // 2, (i_ - i) // 2 + (i_ - i) % 2) for i_, i in zip(shape, input_array.shape)]
#     input_array = np.pad(input_array, shapes, mode='constant')
#     input_array = pad2shape(input_array)
#     # new_image = nib.Nifti1Image(input_array, img.affine)
#     brainmask_path = masks_dir + '/brain_mask_a.nii.gz'
#     nib.save(new_image, brainmask_path)

#     # finds first and last brain coordinates in eah axis to crop from
#     des_shape = [170, 186, 170]
    
#     x, y, z = np.nonzero(input_array)
#     x_s, x_e, x_len = x.min(), x.max(), input_array.shape[0]
#     y_s, y_e, y_len = y.min(), y.max(), input_array.shape[1]
#     z_s, z_e, z_len = z.min(), z.max(), input_array.shape[2]
#     # defines starting coordinates for each axis and stores it in 'mins' list
#     start_vec = (np.ceil((x_e - x_s) / 2) + x_s - des_shape[0] / 2, np.ceil((y_e - y_s) / 2) + y_s - des_shape[1] / 2,
#                  np.ceil((z_e - z_s) / 2) + z_s - des_shape[2] / 2)
#     mins = [0, 0, 0, 0]
#     for i in range(3):
#         if img.shape[i] > des_shape[i]:
#             mins[i] = start_vec[i]

#     # defines exract_ROI parameters and runs it
#     roi = ExtractROI()
    # roi.inputs.in_file = brainmask_path
    # roi.inputs.x_min = int(start_vec[0])
    # roi.inputs.y_min = int(start_vec[1])
    # roi.inputs.z_min = int(start_vec[2])
    # roi.inputs.x_size = des_shape[0]
    # roi.inputs.y_size = des_shape[1]
    # roi.inputs.z_size = des_shape[2]
    # out_path = masks_dir + '/brain_mask_b.nii.gz'
    # # out_path = os.path.join(os.getcwd(), "brain_mask_b.nii.gz") 
    # roi.inputs.roi_file = out_path
    # roi.inputs.t_min = mins[3]
    # roi.inputs.t_size = -1
    # roi.run()

    # # save as a np array
    # img = nib.load(out_path).get_fdata().T
    # out_path = masks_dir + '/brain_mask.npy'
    # # out_path = os.path.join(os.getcwd(), "brain_mask.npy") 
    # np.save(out_path, img)

def sliceSaveComplex(inputImg):
    # saves output brain slices in each axis in Slices/Complexed directory, creates relevant
    # directories when run begins
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    # defines output directories and creates them if necessary
    slices_dir = inputImg[:inputImg.index('Preprocess_Pipeline')] + 'Slices/'
    slices_complexed_dir = slices_dir + 'Complexed/'
    slices_simple_dir = slices_dir + 'Simple/'
    if not os.path.isdir(slices_dir):
        os.mkdir(slices_dir)
        os.mkdir(slices_complexed_dir)
        os.mkdir(slices_simple_dir)
    while not os.path.isdir(slices_dir):
        pass
    # extracts subject name
    sub_id = inputImg.split('/')[-3][12:]
    # loads inputImage
    vol = np.load(inputImg)
    # defines output directory for each axis
    axes = ['xSlice/', 'ySlice/', 'zSlice/']
    # finds middle slices coordinates
    slices = [vol[int(vol.shape[0] / 2), :, :], vol[:, int(vol.shape[1] / 2), :], vol[:, :, int(vol.shape[2] / 2)]]
    # saves middle slices for each axis for each axis output directory
    for i in range(3):
        curr_axis_dir = slices_complexed_dir + axes[i]
        if not os.path.isdir(curr_axis_dir):
            os.mkdir(curr_axis_dir)
        out_save = curr_axis_dir + str(sub_id) + '.png'
        plt.imsave(out_save, slices[i])

    return inputImg[:inputImg.index(inputImg.split('/')[-2])]


def sliceSaveSimple(inputImg):
    # saves output brain slices in each axis in Slices/Simple directory, creates relevant
    # directories when run begins
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    # defines output directories and creates them if necessary
    slices_dir = inputImg[:inputImg.index('Preprocess_Pipeline')] + 'Slices/'
    slices_complexed_dir = slices_dir + 'Complexed/'
    slices_simple_dir = slices_dir + 'Simple/'
    if not os.path.isdir(slices_dir):
        os.mkdir(slices_dir)
        os.mkdir(slices_complexed_dir)
        os.mkdir(slices_simple_dir)

    while not os.path.isdir(slices_dir):
        pass
    # extracts subject name
    sub_id = inputImg.split('/')[-3][12:]
    # loads inputImage
    vol = np.load(inputImg)
    # defines output directory for each axis
    axes = ['xSlice/', 'ySlice/', 'zSlice/']
    # finds middle slices coordinates
    slices = [vol[int(vol.shape[0] / 2), :, :], vol[:, int(vol.shape[1] / 2), :], vol[:, :, int(vol.shape[2] / 2)]]
    # saves middle slices for each axis for each axis output directory
    for i in range(3):
        curr_axis_dir = slices_simple_dir + axes[i]
        if not os.path.isdir(curr_axis_dir):
            os.mkdir(curr_axis_dir)
        out_save = curr_axis_dir + str(sub_id) + '.png'
        plt.imsave(out_save, slices[i])


def saveImageAsNifty(img_array, img_affine, sub_id, funcName):
    # saves a numpy array to nii file
    import nibabel as nib
    import os
    new_image = nib.Nifti1Image(img_array, img_affine)
    out_path = os.path.join(os.getcwd(), funcName + str(sub_id) + '.nii.gz')
    nib.save(new_image, out_path)
    return out_path


def extract_ROI(inputImg):
    # extracts relevant brain area centered and cropped to desired shape, used in Brain_Age_Pipeline_V2
    from nipype.interfaces.fsl import ExtractROI
    import numpy as np
    import nibabel as nib
    import os
    # extracts subject's name, and defines function's name
    sub_id = inputImg.split('/')[-3][12:]
    funcName = 'extract_ROI_'
    # desired crop shape
    des_shape = [90, 120, 99]
    des_shape = [212, 256, 256]
    des_shape = [176, 224, 176]    # this is the data size from Tal
    des_shape = [160, 192, 160]
    des_shape = [160, 176, 160]
    des_shape = [170, 186, 170]

    img = nib.load(inputImg)
    img_ar = img.get_data()
    # finds first and last brain coordinates in eah axis to crop from
    x, y, z = np.nonzero(img_ar)
    x_s, x_e, x_len = x.min(), x.max(), img_ar.shape[0]
    y_s, y_e, y_len = y.min(), y.max(), img_ar.shape[1]
    z_s, z_e, z_len = z.min(), z.max(), img_ar.shape[2]
    # defines starting coordinates for each axis and stores it in 'mins' list
    start_vec = (np.ceil((x_e - x_s) / 2) + x_s - des_shape[0] / 2, np.ceil((y_e - y_s) / 2) + y_s - des_shape[1] / 2,
                 np.ceil((z_e - z_s) / 2) + z_s - des_shape[2] / 2)
    mins = [0, 0, 0, 0]
    for i in range(3):
        if img.shape[i] > des_shape[i]:
            mins[i] = start_vec[i]

    # defines exract_ROI parameters and runs it
    roi = ExtractROI()
    roi.inputs.in_file = inputImg
    roi.inputs.x_min = int(start_vec[0])
    roi.inputs.y_min = int(start_vec[1])
    roi.inputs.z_min = int(start_vec[2])
    roi.inputs.x_size = des_shape[0]
    roi.inputs.y_size = des_shape[1]
    roi.inputs.z_size = des_shape[2]
    out_path = os.path.join(os.getcwd(), funcName + str(sub_id) + '.nii.gz')
    roi.inputs.roi_file = out_path
    roi.inputs.t_min = mins[3]
    roi.inputs.t_size = -1
    roi.run()
    return out_path


def delete_save_trans(sub_dir, anat_path):
    # Functions that saves transformations and shapes from pre chosen steps.
    # The function also deletes heavy memory files to decrease pipeline output memory usage
    import os
    import pickle
    import nibabel as nib
    import shutil

    # extracts subject's name
    sub_id = sub_dir.split('/')[-2]

    # output dir to save affines and shapes in
    out_dir = sub_dir[:sub_dir.index('Preprocess_Pipeline')] + 'Affines_Shapes/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Chosen steps for saving affines and shapes, Also saves a copy of the Anatomical scan for future regestration
    affines_shapes = ['Anat', 'extract_ROI', 'rescaleImage', 'robustFov', 'resampleD', 'deOblique']
    # chosen steps to delete pipeline output fles from
    delete_dir = ['robustFov', 'resampleD', 'deOblique']

    for dir in affines_shapes:
        save_dir = out_dir + dir + '/'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # saves a copy of the anatomical scan
        if dir == 'Anat':
            if os.path.isfile(anat_path):
                os.mkdir(save_dir + sub_id[12:])
                shutil.copyfile(anat_path, save_dir + sub_id[12:] + '/Anat.nii.gz')
            else:
                pass
        else:
            # saves affine transformation and shape for chosen scan in pickle form
            scan_dir = sub_dir + dir + '/'
            scan = [fn for fn in os.listdir(scan_dir) if (scan_dir + fn).endswith('.gz', -3)][0]
            scan_path = scan_dir + scan
            affine_save = save_dir + 'Affines/'
            shape_save = save_dir + 'Shapes/'
            if not os.path.isdir(affine_save):
                os.mkdir(affine_save)
                os.mkdir(shape_save)
            scan_nib = nib.load(scan_path)
            with open(affine_save + sub_id[12:] + '_affine_transformation.pickle', 'wb') as f:
                pickle.dump(scan_nib.affine, f)
            with open(shape_save + sub_id[12:] + '_shape.pickle', 'wb') as f:
                pickle.dump(scan_nib.shape, f)
            # deletes chosen scan files
            if dir in delete_dir:
                os.remove(scan_path)


def delete_save_trans_V2(sub_dir, anat_path):
    # Functions that saves transformations and shapes from pre chosen steps.
    # The function also deletes heavy memory files to decrease pipeline output memory usage
    import os
    import pickle
    import nibabel as nib
    import shutil

    # extracts subject's name
    sub_id = sub_dir.split('/')[-2].split('_')[-1]

    # output dir to save affines and shapes in
    out_dir = sub_dir[:sub_dir.index('Preprocess_Pipeline')] + 'Affines_Shapes/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Chosen steps for saving affines and shapes, Also saves a copy of the Anatomical scan for future regestration
    affines_shapes = ['Anat', 'extract_ROI', 'rescaleImage', 'robustFov', 'resampleD', 'deOblique']
    # chosen steps to delete pipeline output fles from
    delete_dir = ['robustFov', 'resampleD', 'deOblique']

    for dir in affines_shapes:
        save_dir = out_dir + dir + '/'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # saves a copy of the anatomical scan
        if dir == 'Anat':
            if os.path.isfile(anat_path):
                save_sub_dir = save_dir + sub_id + '/'
                os.mkdir(save_sub_dir)
                with open(save_sub_dir + sub_id + '_anat_location.pickle', 'wb') as f:
                    pickle.dump(anat_path, f)
            else:
                pass
        else:
            # saves affine transformation and shape for chosen scan in pickle form
            scan_dir = sub_dir + dir + '/'
            scan = [fn for fn in os.listdir(scan_dir) if
                    (scan_dir + fn).endswith('.gz', -3) or (scan_dir + fn).endswith('.nii', -4)][0]
            scan_path = scan_dir + scan
            affine_save = save_dir + 'Affines/'
            shape_save = save_dir + 'Shapes/'
            if not os.path.isdir(affine_save):
                os.mkdir(affine_save)
                os.mkdir(shape_save)
            scan_nib = nib.load(scan_path)
            with open(affine_save + sub_id + '_affine_transformation.pickle', 'wb') as f:
                pickle.dump(scan_nib.affine, f)
            with open(shape_save + sub_id + '_shape.pickle', 'wb') as f:
                pickle.dump(scan_nib.shape, f)
            # deletes chosen scan files
            if dir in delete_dir:
                os.remove(scan_path)


if __name__ == "__main__":
    delete_save_trans_V2(
        '/media/galia_nas/workingdir/ofek/filtered_diet_data/pipeline_output_t0/Preprocess_Pipeline_V2/_subject_id_s27/',
        '/media/galia_nas/workingdir/ofek/filtered_diet_data/t0/s27/HASAN^AMNON_JAK_T1_3D_TFE_SENSE_3_1.nii')





