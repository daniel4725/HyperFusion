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
