#%%
import numpy as np
import monai
import torch



# https://docs.monai.io/en/stable/transforms.html
tform_dict = {"None": None, None: None}  # both forms of None must have None value
deterministic = False


# ------------------------------------------
tform_name = "normalize"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
#  write your augmentation below:
    monai.transforms.NormalizeIntensity(nonzero=True)  
])
# ------------------------------------------
tform_name = "hippo_crop"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
#  write your augmentation below:
    lambda img: img[:, 25: 25 + 64, 55: 55 + 96, 88: 88 + 64],
    monai.transforms.NormalizeIntensity(nonzero=True)
])
# ------------------------------------------
tform_name = "hippo_crop_2sides"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
#  write your augmentation below:
    lambda img: img[:, 25: 25 + 64, 55: 55 + 96, 85 - 64: 85 + 64],
#     lambda img: img[:, 35: 35 + 64, 55: 55 + 96, 85 - 64: 85 + 64],
#     lambda img: img[:, 50: 50 + 64, 55: 55 + 96, 85 - 64: 85 + 64],
    monai.transforms.NormalizeIntensity(nonzero=True)  
])
# ------------------------------------------
tform_name = "hippo_crop_lNr"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
#  write your augmentation below:
    monai.transforms.RandFlip(prob=0.5, spatial_axis=2),  # left brain to right
    lambda img: img[:, 25: 25 + 64, 55: 55 + 96, 85: 85 + 64],
    # lambda img: img[:, 35: 35 + 64, 55: 55 + 96, 85: 85 + 64],
    # lambda img: img[:, 50: 50 + 64, 55: 55 + 96, 85: 85 + 64],
    monai.transforms.NormalizeIntensity(nonzero=True)
])

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------  l2r transforms -------------------------------------------
# ---------------------------------------------------------------------------------------------------
tform_name = "hippo_crop_2sides_for_load_2_ram_func"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = lambda img: img[:, 25: 25 + 64, 55: 55 + 96, 85 - 64: 85 + 64]
#  write your augmentation below:
#     lambda img: img[:, 50: 50 + 64, 55: 55 + 96, 85 - 64: 85 + 64]
#     lambda img: img[:, 25: 25 + 64, 55: 55 + 96, 85 - 64: 85 + 64]



# ------------------------------------------
tform_name = "hippo_crop_lNr_l2r"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
    monai.transforms.RandFlip(prob=0.5, spatial_axis=2),  # left brain to right
    lambda img: img[:, :, :, 64:],
    monai.transforms.NormalizeIntensity(nonzero=True)
])
# ------------------------------------------
tform_name = "hippo_crop_lNr_affine_l2r"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
    monai.transforms.RandFlip(prob=0.5, spatial_axis=2),  # left brain to right
    monai.transforms.RandAffine(
        prob=0.4,
        rotate_range=[np.deg2rad(1)] * 3,   # * 3 is in each channel
        scale_range=[0.001] * 3,
        padding_mode='zeros',
        mode='bilinear'
    ),
    lambda img: img[:, :, :, 64:],
    monai.transforms.NormalizeIntensity(nonzero=True)
])
# ------------------------------------------
tform_name = "hippo_crop_lNr_noise_scale_l2r"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
    monai.transforms.RandFlip(prob=0.5, spatial_axis=2),  # left brain to right
    monai.transforms.RandAffine(
        prob=0.5,
        scale_range=[0.001] * 3,
        padding_mode='zeros',
        mode='bilinear'
    ),
    lambda img: img[:, :, :, 64:],
    monai.transforms.NormalizeIntensity(nonzero=True),
    monai.transforms.RandGaussianNoise(prob=0.3, mean=0.0, std=0.01)
])
# ------------------------------------------
tform_name = "hippo_crop_lNr_noise_m0s1_l2r"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
    monai.transforms.RandFlip(prob=0.5, spatial_axis=2),  # left brain to right
    lambda img: img[:, :, :, 64:],
    monai.transforms.NormalizeIntensity(nonzero=True),
    monai.transforms.RandGaussianNoise(prob=0.5, mean=0.0, std=1)
])
# ------------------------------------------
tform_name = "hippo_crop_lNr_noise_affine_l2r"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
    monai.transforms.RandFlip(prob=0.5, spatial_axis=2),  # left brain to right
    monai.transforms.RandAffine(
        prob=0.5,
        rotate_range=[np.deg2rad(0.5)] * 3,  # * 3 is in each channel
        scale_range=[0.001] * 3,
        padding_mode='zeros',
        mode='bilinear'
    ),
    lambda img: img[:, :, :, 64:],
    monai.transforms.NormalizeIntensity(nonzero=True),
    monai.transforms.RandGaussianNoise(prob=0.5, mean=0.0, std=0.01)
])












tform_name = "basic_aug"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
#  write your augmentation below:
    # monai.transforms.RandFlip(prob=0.1, spatial_axis=2),  # left brain to right
    monai.transforms.RandAffine(
                prob=0.3,
                rotate_range=[np.deg2rad(4)] * 3,  
                scale_range=[0.05] * 3,
                padding_mode='zeros',
                mode='bilinear'
                ),
    monai.transforms.NormalizeIntensity(nonzero=True)
    # monai.transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.05),
])
# ------------------------------------------
tform_name = "augment1"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
#  write your augmentation below:
    monai.transforms.RandFlip(prob=0.3, spatial_axis=2),  # left brain to right
    monai.transforms.RandAffine(
                prob=0.3,
                rotate_range=6*(2*np.pi/360),  # 6 degrees
                scale_range=0.1,
                shear_range=0.06,
                padding_mode='zeros',
                mode='bilinear'
                ),
    monai.transforms.NormalizeIntensity(nonzero=True),  # TODO maybe check with nonzero=True 
    monai.transforms.RandGaussianNoise(prob=0.3, mean=0.0, std=0.1),
])
# ------------------------------------------
tform_name = "augment2"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
#  write your augmentation below:
    monai.transforms.NormalizeIntensity(nonzero=True),  # TODO maybe check with nonzero=True 
    monai.transforms.RandGaussianNoise(prob=1, mean=0.0, std=0.1)
])
# ------------------------------------------
tform_name = "augment3"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
#  write your augmentation below:
])
# ------------------------------------------



# ------------------------------------------
tform_name = "hippo_crop_lNr_l2r_tst"
# assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
    lambda img: img[:, :, :, 64:],
    monai.transforms.NormalizeIntensity(nonzero=True)
])
# ------------------------------------------
tform_name = "hippo_crop_lNr_tst"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
#  write your augmentation below:
    lambda img: img[:, 25: 25 + 64, 55: 55 + 96, 88: 88 + 64],
    monai.transforms.NormalizeIntensity(nonzero=True)
])

# ---------------------------------------------------------------------------------------------------
# ------------------------------------  resize ablation transforms ----------------------------------
# ---------------------------------------------------------------------------------------------------
# ------------------------------------------
tform_name = "brain_resize_2sides_for_load_2_ram_func_train"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Resize((64, 96, 128))

# ------------------------------------------
tform_name = "brain_resize_2sides_for_load_2_ram_func_val"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
    monai.transforms.Resize((64, 96, 128)),
    monai.transforms.NormalizeIntensity(nonzero=True)
])

# ------------------------------------------
tform_name = "crop_lNr_after_l2r"
assert tform_name not in tform_dict.keys()
tform_dict[tform_name] = monai.transforms.Compose([
    monai.transforms.RandFlip(prob=0.5, spatial_axis=2),  # left brain to right
    lambda img: img[:, :, :, 64:],
    monai.transforms.NormalizeIntensity(nonzero=True)
])


if __name__=="__main__":
    from data_handler import get_dataloaders
    import matplotlib.pyplot as plt
    ADNI_dir = "/usr/local/faststorage/adni_class_pred_1x1x1_v1"
    metadata_path = "metadata_by_features_sets/set-1.csv"
    data_fold = 0
    transform = tform_dict["hippo_crop"]

    loaders = get_dataloaders(batch_size=5, adni_dir=ADNI_dir,
                             metadata_path=metadata_path,
                             fold=data_fold, transform=transform)
    train_loader, valid_loader = loaders

    train_ds = train_loader.dataset
    valid_ds = valid_loader.dataset
    img, tabular, y = train_ds.__getitem__(59)
    plt.imshow(img[0, 45])
