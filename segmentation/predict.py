#%% Imports

import os
import shutil
import subprocess
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
)

#%% Yolov5 instance segmentation prediction

# clone yolov5 repo
base_path = os.path.dirname(os.path.dirname(__file__))
clone_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/yolov5")
Repo.clone_from("https://github.com/ultralytics/yolov5.git", clone_path)

#%%
# directories and params for yolo prediction
base_path = os.path.dirname(os.path.dirname(__file__))
clone_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/yolov5")
path = os.getcwd()
os.chdir(clone_path + "/segment")
weights_fdr = base_path + "/models/yolov5_best_model/best.pt"
source_fdr = base_path + "/data/testset/imgs_scraped_clean"
project_fdr = base_path + "/data/testset/imgs_original"
# conf_fdr = "0.4"  # can be adjusted to desired confidence level
if os.path.exists(base_path + "/data/testset/imgs_original"):
    shutil.rmtree(base_path + "/data/testset/imgs_original")

#%%
import wandb

run = wandb.init()
artifact = run.use_artifact(
    "nail_project/YOLOv5-Segment/run_pd58lz0y_model:v299", type="model"
)
artifact_dir = artifact.download()
#%%

# conf_fdr = "0.067"  # for area affected model
conf_fdr = "0.676"  # for whole nail model
# conf_fdr = "0.2"  # for whole nail model: also trying lower bc 2 false negatives; doesnt work bc of false positives!

# iou_fdr = "1.0" # for area affected model
iou_fdr = "0.6"  # for whole nail model

line_thick = "1"  # can be adjusted to desired line thickness
# data_fdr = base_path + "/models/yolov5/Onychomycosis-segmentation-4/data.yaml"

#%%

subprocess.run(
    [
        "python3",
        "predict.py",
        "--weights",
        weights_fdr,
        #        "--data",
        #        data_fdr,
        "--line-thickness",
        line_thick,
        "--source",
        source_fdr,
        #        "--save-txt",
        "--name",
        project_fdr,
        "--project",
        project_fdr,
        "--iou-thres",
        iou_fdr,
        # "--data",
        # data_fdr,
        "--conf-thres",
        conf_fdr,
        "--save-txt",
    ]
)
os.chdir(path)


# clone_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/yolov5")
# os.chdir(clone_path + "/segment")
# source_fdr = "0"
# subprocess.run(
#     [
#         "python3",
#         "predict.py",
#         "--weights",
#         weights_fdr,
#         "--source",
#         source_fdr,
#     ]
# )
# os.chdir(path)

#%% Crop images and update segmentation mask
# TODO: Find the right value for max_extra_pad, see below

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
