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

from utils.delete_images import (
    delete_small_images,
    delete_extreme_aspect_ratio_images,
    delete_duplicate_images,
    delete_txt_files_for_del_images,
)

from utils.read_show_crop_imgs import (
    read_image_label,
    crop_image_label,
    save_image_label,
)

#%% Scrape images from google

path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data/testset/imgs_scraped"
)

scrape_images(
    path=path,
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
    max_n=30,  # max number of images to scrape per search term
)
#%% Some preprocessing

# rename images according to the subfolders they are in
rename_scraped_images(path=path)

# copy all images from subfolders to a single folder imgs_scraped_clean
copy_all_imgs_to_one_folder(path_old=path, path_new=path + "_clean")

# delete very small images
delete_small_images(
    path=path + "_clean",
    min_width=85,
    min_height=85,
)

# delete images with extreme aspect ratio
delete_extreme_aspect_ratio_images(
    path=path + "_clean",
    max_aspect_ratio=2.4,
    min_aspect_ratio=0.417,
)

# delete duplicates using embeddings and cosine similarity
# TODO: this is quite slow, maybe use a faster method
delete_duplicate_images(path=path + "_clean")

#%% Yolov5 instance segmentation for nail cropping

# clone yolov5 repo
base_path = os.path.dirname(os.path.dirname(__file__))
clone_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/yolov5")
Repo.clone_from("https://github.com/ultralytics/yolov5.git", clone_path)

# directories and params for yolo segmentation
path = os.getcwd()
os.chdir(clone_path + "/segment")
source_fdr = base_path + "/data/testset/imgs_scraped_clean"
project_fdr = base_path + "/data/testset/imgs_original"
if os.path.exists(base_path + "/data/testset/imgs_original"):
    shutil.rmtree(base_path + "/data/testset/imgs_original")
weights_fdr = (
    base_path + "/models/yolov5_best_model/best.pt"
)  # path to best nail segmentation model
conf_fdr = "0.4"

# run yolov5 prediction on scraped images
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

#%% Crop images based on yolov5 prediction, and update segmentation mask coordinates

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

    # get number of objects (nails) in the image
    with open(path_txt_original + txt_file, "r") as f:
        nobj_in_image = len(f.readlines())

    # save one cropped image and txt file per object (nail) in image
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
