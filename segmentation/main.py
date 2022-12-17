#%% Imports

import os
import shutil
import subprocess
import numpy as np
from git.repo.base import Repo

from utils.scrape_images import (
    scrape_images,
    rename_scraped_images,
    copy_all_imgs_to_one_folder,
)

from utils.delete_duplicates import (
    delete_small_images,
    delete_duplicate_images,
    delete_extreme_aspect_ratio_images,
    delete_txt_files_for_del_images,
)

from utils.read_show_crop_imgs import (
    read_image_label,
    crop_image_label,
    save_image_label,
    get_image_mask,
)

#%% Scrape images from google

# Search term(s) are provided as input list(s). If multiple search terms are
# provided per list, the first search term will be used as the representative
# name of the class and the others as synonyms for this class.

scrape_images(
    dir_name="data/testset/imgs_scraped",
    searches=[
        [
            "Nagel gesund",  # german
            "Fingernagel",  # german
            "Fußnagel",  # german
            "ongle sain",  # french
            "unghia sana",  # italian
            "nagel gezond",  # dutch
            "sunde negle",  # danish
            "zdrowy paznokieć",  # polish # czech
            "nehty zdravé",  # czech
            "uñas sanas",  # spanish
        ],
        [
            "Onychomykose Nagel",  # german
            "Nagelmykose",  # german
            "Nagelpilz",  # german
            "mycose des ongles",  # french
            "fungo delle unghie",  # italian
            "nagelschimmel",  # dutch
            "neglesvamp",  # danish
            "Grzyb paznokcia",  # polish
            "Onychomycóza nehtů",  # czech
            "onicomicosis",  # spanish
        ],
        [
            "Dystrophie Nagel",  # german
            "Nageldystrophie",  # german
            "Onychodystrophie",  # german
            "dystrophie des ongles",  # french
            "distrofia ungueale",  # italian
            "nagel dystrofie",  # dutch
            "Negledystrofi",  # danish
            "dystrofia paznokci",  # polish
            "Dystrofie nehtů",  # czech
            "distrofia ungueal",  # spanish
        ],
        [
            "Melanonychie Nagel",  # german
            "Streifenförmige Nagelpigmentierung",  # german
            "Longitudinale Melanonychie",  # german
            "mélanonychie",  # french
            "melanonichia",  # italian
            "Melanonychia-nagel",  # dutch
            "melanonychia negl",  # danish
            "melanonychia paznokcia",  # polish
            "melanonychie",  # czech
            "melanoniquia",  # spanish
        ],
        [
            "Onycholyse Nagel",  # german
            "Nagelablösung",  # german
            "Nagelabhebung",  # german
            "Ongle décollement",  # french
            "unghia onicolisi",  # italian
            "loslating van de nagel",  # dutch
            "onykolyse",  # danish
            "onycholiza paznokcia",  # polish
            "onycholýza",  # czech
            "onicólisis",  # spanish
        ],
    ],
    max_n=30,
)
#%% rename images according to the subfolders they are in. removing broken
# images that cannot be opened
rename_scraped_images(dir_name="data/testset/imgs_scraped")
#%%
# copy all images from subfolders to a single folder imgs_scraped_clean
copy_all_imgs_to_one_folder(new_folder="data/testset/imgs_scraped_clean")

#%%
# delete very small images
delete_small_images(
    min_width=85, min_height=85, dir_name="data/testset/imgs_scraped_clean"
)

# delete images with extreme aspect ratio
delete_extreme_aspect_ratio_images(
    max_aspect_ratio=2.4,
    min_aspect_ratio=0.417,
    dir_name="data/testset/imgs_scraped_clean",
)

#%% delete duplicates using embeddings and cosine similarity

# TODO: this is quite slow, maybe use a faster method
delete_duplicate_images(dir_name="data/testset/imgs_scraped_clean")

#%% Yolov5 instance segmentation prediction

# clone yolov5 repo
base_path = os.path.dirname(os.path.dirname(__file__))
clone_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/yolov5")
Repo.clone_from("https://github.com/ultralytics/yolov5.git", clone_path)

# directories and params for yolo prediction
path = os.getcwd()
os.chdir(clone_path + "/segment")
weights_fdr = base_path + "/models/yolov5_best_model/best.pt"
source_fdr = base_path + "/data/testset/imgs_scraped_clean"
project_fdr = base_path + "/data/testset/imgs_original"
conf_fdr = "0.4"  # can be adjusted to desired confidence level
if os.path.exists(base_path + "/data/testset/imgs_original"):
    shutil.rmtree(base_path + "/data/testset/imgs_original")

# run yolov5 prediction
subprocess.run(
    [
        "python3",
        "predict.py",
        "--weights",
        weights_fdr,
        "--source",
        source_fdr,
        "--save-txt",
        "--name",
        project_fdr,
        "--project",
        project_fdr,
        "--conf",
        conf_fdr,
        "--save-txt",
    ]
)
os.chdir(path)

#%% Crop images and update segmentation mask

path_imgs_original = os.path.join(base_path, "data/testset/imgs_scraped_clean/")
path_txt_original = os.path.join(base_path, "data/testset/imgs_original", "labels/")
path_imgs_cropped = os.path.join(base_path, "data/testset/imgs_cropped/")
path_txt_cropped = os.path.join(base_path, "data/testset/txt_cropped/")

txt_original = os.listdir(path_txt_original)

# go through each original txt file containing segmentation mask coordinates
# (nail), load the corresponding image file, crop the image based on the mask,
# update txt coordinates wrt new image, and save everything.
for txt_file in txt_original:

    # image file name is the same as txt file name
    img_file = txt_file.split(".")[0] + ".jpg"

    # number of lines is the number of objects (nails) in the image
    with open(path_txt_original + txt_file, "r") as f:
        nobj_in_image = len(f.readlines())

    # some images have more than one object (nail) - save one cropped image and
    # txt file per nail
    for cur_obj in range(nobj_in_image):

        image, polygon_nail, obj_class = read_image_label(
            path_imgs_original + img_file, path_txt_original + txt_file, cur_obj
        )

        # crop image and update segmentation mask based on cropped image. Set
        # desired extra padding in pixels
        (
            image_cropped,
            polygon_nail_cropped,
            polygon_nail_cropped_norm,
        ) = crop_image_label(
            image, polygon_nail, square=False, max_extra_pad_prop=0.2, obj_class=0
        )

        # new file names for saved cropped image and txt file with updated
        # segmentation mask cur_obj is added to the file name to distinguish
        # between multiple objects in the same image
        img_file_save = img_file.split(".")[0] + "_" + str(cur_obj) + ".jpg"
        txt_file_save = img_file.split(".")[0] + "_" + str(cur_obj) + ".txt"

        save_image_label(
            image_cropped,
            polygon_nail_cropped_norm,
            path_imgs_cropped,
            path_txt_cropped,
            img_file_save,
            txt_file_save,
        )

#%% Delete very small crop images and corresponding txt files

# TODO: set min_width and min_height to desired values

del_imagenames = delete_small_images(
    min_width=80,
    min_height=80,
    dir_name="data/testset/imgs_cropped",
    return_del_filenames=True,
)

delete_txt_files_for_del_images(
    file_names_to_delete=del_imagenames,
    dir_name="data/testset/txt_cropped",
    return_del_filenames=False,
)

#%% Area affected segmentation GROUND TRUTH: determine proportion of nail
# affected by fungus in ground truth validation images

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
    proportion_affected = np.sum(
        area_gt_mask_combined[area_gt_mask_combined > 0]
    ) / np.sum(nail_gt_mask[nail_gt_mask > 0])

    dict_prop_affected.update({img_file: proportion_affected})

# average proportion of nail affected by fungus in ground truth validation images
prop_affected_list = list(dict_prop_affected.values())

# descriptive statistics of prop_affected_list
print("Mean: ", np.mean(prop_affected_list))
print("Median: ", np.median(prop_affected_list))
print("Std: ", np.std(prop_affected_list))
print("Min: ", np.min(prop_affected_list))
print("Max: ", np.max(prop_affected_list))

import matplotlib.pyplot as plt

# plot histogram of prop_affected_list
plt.hist(prop_affected_list, bins=20)


# %% Plotting ground truth validation images area affected
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(22, 18))
axes = fig.subplots(nrows=1, ncols=2)

polygon_area_gt_all_areas = area_gt_mask[0] + area_gt_mask[0]
axes[0].imshow(polygon_area_gt_all_areas)

#%% Area affected segmentation PREDICTIONS: determine proportion of nail
# affected by fungus in predictions for validation images


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

# TODO: currently I put the ground truth txt files in the prediction folder as we don't have
# the predictions yet, so need to change this later
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
dict_dice_score = {}

# looping over ground truth txt files bc in prediction some txt files are missing due to False Negatives
for txt_file_pred in files_val_txt_area_gt:  # files_val_txt_area_pred

    # image file name is the same as txt file name
    img_file_pred = txt_file_pred[:-4] + ".jpg"

    # if txt file for affected area exists (i.e., not a False negative), then
    # proportion of nail affected is calculated
    try:
        # number of lines is the number of objects (nails) in the image
        with open(path_val_txt_area_pred + txt_file_pred, "r") as f:
            nareas_affected_pred = len(f.readlines())

        # assuming there is only one nail per image
        imag_nail_pred, polygon_nail_pred, obj_class_pred = read_image_label(
            path_val_imgs_nail_pred + img_file_pred,
            path_val_txt_nail_pred + txt_file_pred,
            txt_row_obj=0,
        )
        nail_pred_mask = get_image_mask(imag_nail_pred, polygon_nail_pred)

        area_gt_mask_pred = []
        for cur_area_pred in range(nareas_affected_pred):

            imag_area_pred, polygon_area_pred, obj_class_pred = read_image_label(
                path_val_imgs_area_pred + img_file_pred,
                path_val_txt_area_pred + txt_file_pred,
                cur_area_pred,
            )
            area_gt_mask_pred.append(get_image_mask(imag_area_pred, polygon_area_pred))

        # combined version of all ground truth area affected masks
        area_gt_mask_pred_combined = np.sum(area_gt_mask_pred, axis=0)

        # ground truth propotion of whole nail affected
        proportion_affected = np.sum(
            area_gt_mask_pred_combined[area_gt_mask_pred_combined > 0]
        ) / np.sum(nail_pred_mask[nail_pred_mask > 0])

        dict_prop_affected_pred.update({img_file_pred: proportion_affected})

        # calculate DICE coefficient
        dice_score = DICE_COE(nail_pred_mask, area_gt_mask_pred_combined)
        dict_dice_score.update({img_file_pred: dice_score})

    except:
        # if False negative, then proportion of nail affected and dice score is 0
        proportion_affected = 0
        dice_score = 0
        dict_prop_affected_pred.update({img_file_pred: proportion_affected})
        dict_dice_score.update({img_file_pred: dice_score})
        print("File not found bc Fale Negative: ", txt_file_pred)

# average proportion of nail affected by fungus in ground truth validation images
prop_affected_list_pred = list(dict_prop_affected_pred.values())

# descriptive statistics of prop_affected_list_pred
print("Mean: ", np.mean(prop_affected_list_pred))
print("Median: ", np.median(prop_affected_list_pred))
print("Std: ", np.std(prop_affected_list_pred))
print("Min: ", np.min(prop_affected_list_pred))
print("Max: ", np.max(prop_affected_list_pred))

import matplotlib.pyplot as plt

# plot histogram of prop_affected_list_pred
plt.hist(prop_affected_list_pred, bins=20)

# %% Difference between ground truth and prediction per image
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

fig = plt.figure(figsize=(18, 10))
axes = fig.subplots(nrows=1, ncols=3)

axes[0].hist(np.array(prop_affected_list) - np.array(prop_affected_list_pred), bins=20)
axes[1].hist(
    np.abs((np.array(prop_affected_list) - np.array(prop_affected_list_pred))), bins=20
)

mean_absolute_error_ = mean_absolute_error(prop_affected_list, prop_affected_list_pred)

# plot mean absolute error_ on histogram
axes[2].bar(
    mean_absolute_error_,
    align="center",
    height=mean_absolute_error_,
)
axes[2].set_xticklabels([])

# add title and axis names
axes[0].set_title("Difference GT and PRED prop affected per image")
axes[1].set_title("Abs Difference GT and PRED prop affected per image")
axes[2].set_title("Mean Absolute Error")
