import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from utils.read_show_crop_imgs import (
    read_image_label,
    get_image_mask,
)

#%% Area affected segmentation GROUND TRUTH: determine proportion of nail
# affected by fungus in ground truth validation images (ground truth area
# affected, ground truth nail)

base_path = os.path.dirname(os.path.dirname(__file__))

path_val_imgs_nail_gt = os.path.join(
    base_path, "data/valset/segmentation/ground_truth/nail/images/"
)
path_val_txt_nail_gt = os.path.join(
    base_path, "data/valset/segmentation/ground_truth/nail/labels/"
)
path_val_imgs_area_gt = os.path.join(
    base_path, "data/valset/segmentation/ground_truth/area_affected/images/"
)
path_val_txt_area_gt = os.path.join(
    base_path, "data/valset/segmentation/ground_truth/area_affected/labels/"
)

files_val_txt_nail_gt = os.listdir(path_val_txt_nail_gt)
files_val_txt_nail_gt = [f for f in files_val_txt_nail_gt if not f.startswith(".")]

files_val_txt_area_gt = os.listdir(path_val_txt_area_gt)
files_val_txt_area_gt = [f for f in files_val_txt_area_gt if not f.startswith(".")]

dict_prop_affected = {}
dict_area_affected_gt_mask_combined = (
    {}
)  # to later calculate dice score ground truth vs predicted area affected

for txt_file in files_val_txt_area_gt:

    # image file name is the same as txt file name
    img_file = txt_file[:-4] + ".jpg"

    # number of lines is the number of objects (nails) in the image
    with open(path_val_txt_area_gt + txt_file, "r") as f:
        nareas_affected = len(f.readlines())

    # assuming there is only one nail per image
    imag_nail_gt, polygon_nail_gt, obj_class = read_image_label(
        path_val_imgs_nail_gt + img_file, path_val_txt_nail_gt + txt_file, txt_row_obj=0
    )
    nail_gt_mask = get_image_mask(imag_nail_gt, polygon_nail_gt)

    area_gt_mask = []
    for cur_area in range(nareas_affected):

        imag_area_gt, polygon_area_gt, obj_class = read_image_label(
            path_val_imgs_area_gt + img_file,
            path_val_txt_area_gt + txt_file,
            cur_area,
        )
        area_gt_mask.append(get_image_mask(imag_area_gt, polygon_area_gt))

    # combined version of all ground truth area affected masks
    area_gt_mask_combined = np.sum(area_gt_mask, axis=0)

    # ground truth propotion of whole nail affected
    proportion_affected = len(  # corrected bug, this should be len not np.sum
        area_gt_mask_combined[area_gt_mask_combined > 0]
    ) / len(
        nail_gt_mask[nail_gt_mask > 0]
    )  # corrected bug, this should be len not np.sum

    # if ground truth proportion affected is greater than 1 (bigger than the
    # ground truth nail itself), then it is set to 1. This was not observed.
    if proportion_affected > 1:
        proportion_affected = 1

    dict_prop_affected.update({img_file: proportion_affected})

    # save image name and mask to later calculate dice score ground truth vs
    # prediction area affected
    dict_area_affected_gt_mask_combined.update({img_file: area_gt_mask_combined})

# average proportion of nail affected by fungus in ground truth validation
# images
prop_affected_list = list(dict_prop_affected.values())

# descriptive statistics of prop_affected_list
print("Mean: ", np.mean(prop_affected_list))
print("Median: ", np.median(prop_affected_list))
print("Std: ", np.std(prop_affected_list))
print("Min: ", np.min(prop_affected_list))
print("Max: ", np.max(prop_affected_list))

# plot histogram of prop_affected_list
plt.hist(prop_affected_list, bins=20)

# Plotting ground truth validation images area affected
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(22, 18)) axes = fig.subplots(nrows=1, ncols=2)
# polygon_area_gt_all_areas = area_gt_mask[0] + area_gt_mask[0]
# axes[0].imshow(polygon_area_gt_all_areas)

#%% Area affected segmentation PREDICTIONS: determine proportion of nail
# affected by fungus in predictions for validation images (predicted area
# affected, predicted nail)


def DICE_COE(mask1, mask2):
    mask1[mask1 > 0] = 1  # bc mask was 255
    mask2[mask2 > 0] = 1  # bc mask was 255
    intersect = np.sum(mask1 * mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3)  # for easy reading
    return dice


base_path = os.path.dirname(os.path.dirname(__file__))

path_val_imgs_nail_pred = os.path.join(
    base_path, "data/valset/segmentation/prediction/nail/images/"
)

path_val_txt_nail_pred = os.path.join(
    base_path, "data/valset/segmentation/prediction/nail/labels/"
)
path_val_imgs_area_pred = os.path.join(
    base_path, "data/valset/segmentation/prediction/area_affected/images/"
)
path_val_txt_area_pred = os.path.join(
    base_path, "data/valset/segmentation/prediction/area_affected/labels/"
)

files_val_txt_nail_pred = os.listdir(path_val_txt_nail_pred)
files_val_txt_nail_pred = [f for f in files_val_txt_nail_pred if not f.startswith(".")]

files_val_txt_area_pred = os.listdir(path_val_txt_area_pred)
files_val_txt_area_pred = [f for f in files_val_txt_area_pred if not f.startswith(".")]

dict_prop_affected_pred = {}
dict_dice_score = {}  # for dice score predicted area affected vs predicted whole nail
dict_area_affected_mask_pred_combined = (
    {}
)  # to later calculate dice score ground truth vs predicted area affected

# looping over ground truth txt files bc for prediction some txt files are
# missing due to False Negatives
for txt_file_pred in files_val_txt_area_gt:  # files_val_txt_area_pred

    # image file name is the same as txt file name
    img_file_pred = txt_file_pred[:-4] + ".jpg"

    # if txt file for affected area exists (i.e., not a False negative), then
    # proportion of nail affected is calculated
    try:
        # number of lines is the number of objects (nails) in the image
        with open(path_val_txt_area_pred + txt_file_pred, "r") as f:
            nareas_affected_pred = len(f.readlines())

        # assuming there is only one nail per image, taking row 0 (this would be
        # a problem if there were more than one nail, here it is fine)
        imag_nail_pred, polygon_nail_pred, obj_class_pred = read_image_label(
            path_val_imgs_nail_pred + img_file_pred,
            path_val_txt_nail_pred + txt_file_pred,
            txt_row_obj=0,
        )
        nail_pred_mask = get_image_mask(imag_nail_pred, polygon_nail_pred)

        # go through all affected areas in the image
        area_gt_mask_pred = []
        for cur_area_pred in range(nareas_affected_pred):

            imag_area_pred, polygon_area_pred, obj_class_pred = read_image_label(
                path_val_imgs_area_pred + img_file_pred,
                path_val_txt_area_pred + txt_file_pred,
                cur_area_pred,
            )
            area_gt_mask_pred.append(get_image_mask(imag_area_pred, polygon_area_pred))

        # combined version of all predicted area affected masks
        area_gt_mask_pred_combined = np.sum(area_gt_mask_pred, axis=0)

        # predicted propotion of whole nail affected
        proportion_affected = len(  # corrected bug, this should be len not np.sum
            area_gt_mask_pred_combined[area_gt_mask_pred_combined > 0]
        ) / len(
            nail_pred_mask[nail_pred_mask > 0]
        )  # corrected bug, this should be len not np.sum

        # if predicted proportion affected is greater than 1 (bigger than the
        # predicted nail itself), then it is set to 1
        if proportion_affected > 1:
            proportion_affected = 1

        dict_prop_affected_pred.update({img_file_pred: proportion_affected})

        # calculate DICE coefficient for predicted area affected vs predicted
        # whole nail
        dice_score = DICE_COE(nail_pred_mask, area_gt_mask_pred_combined)
        dict_dice_score.update({img_file_pred: dice_score})

        # save image name and mask to later calculate dice score ground truth vs
        # prediction area affected
        dict_area_affected_mask_pred_combined.update(
            {img_file_pred: area_gt_mask_pred_combined}
        )

    except FileNotFoundError:
        # if False negative, then proportion of nail affected and dice score is
        # 0
        proportion_affected = 0
        dice_score = 0
        dict_prop_affected_pred.update({img_file_pred: proportion_affected})
        dict_dice_score.update({img_file_pred: dice_score})

        # save image name and mask to later calculate dice score ground truth vs
        # prediction area affected create a mask of zeros with the same size as
        # the image, bc no prediction
        image_for_dice_mask_zeros = cv2.imread(path_val_imgs_nail_pred + img_file_pred)
        dice_mask_zeros = np.zeros(
            (image_for_dice_mask_zeros.shape[0], image_for_dice_mask_zeros.shape[1]),
            dtype=np.uint8,
        )
        dict_area_affected_mask_pred_combined.update({img_file_pred: dice_mask_zeros})

        print("File not found bc Fale Negative: ", txt_file_pred)

# average proportion of nail affected by fungus in ground truth validation
# images
prop_affected_list_pred = list(dict_prop_affected_pred.values())

# descriptive statistics of prop_affected_list_pred
print("Mean: ", np.mean(prop_affected_list_pred))
print("Median: ", np.median(prop_affected_list_pred))
print("Std: ", np.std(prop_affected_list_pred))
print("Min: ", np.min(prop_affected_list_pred))
print("Max: ", np.max(prop_affected_list_pred))

# plot histogram of prop_affected_list_pred
plt.hist(prop_affected_list_pred, bins=20)

# %% Difference between ground truth and predicted proportion of nail affected
# per image

fig = plt.figure(figsize=(7, 5))
axes = fig.subplots(nrows=1, ncols=2)

axes[0].hist(np.array(prop_affected_list) - np.array(prop_affected_list_pred), bins=20)

mean_absolute_error_ = mean_absolute_error(prop_affected_list, prop_affected_list_pred)

# plot mean absolute error_ on histogram
axes[1].bar(
    mean_absolute_error_,
    align="center",
    height=mean_absolute_error_,
)

# add title in two lines
axes[0].set_title("Difference GT and PRED \n proportion affected per image")
axes[1].set_title("Mean Absolute Error")

# %% DICE coefficient between predicted area affected and ground truth area
# affected

area_affected_pred_vs_GT_dice = {}
for image_name in dict_area_affected_gt_mask_combined:
    dice_score = DICE_COE(
        dict_area_affected_gt_mask_combined[image_name],
        dict_area_affected_mask_pred_combined[image_name],
    )
    area_affected_pred_vs_GT_dice.update({image_name: dice_score})

# average dice score
mean_dice = np.mean(list(area_affected_pred_vs_GT_dice.values()))

# histogram of dice scores
plt.hist(list(area_affected_pred_vs_GT_dice.values()), bins=20)

# plot a histogram of dice scores and mean dice score
fig = plt.figure(figsize=(7, 5))
axes = fig.subplots(nrows=1, ncols=2)

axes[0].hist(list(area_affected_pred_vs_GT_dice.values()), bins=20)
axes[1].bar(
    mean_dice,
    align="center",
    height=mean_dice,
)

axes[0].set_title("Dice coef GT and PRED area")
axes[1].set_title("Mean Dice coef")
