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

path_base = os.path.dirname(os.path.dirname(__file__))
path_imgs_scraped = path_base + "/data/testset/imgs_scraped"

if __name__ == "__main__":

    #%% Scrape images from google

    scrape_images(
        path=path_imgs_scraped,
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
    rename_scraped_images(path=path_imgs_scraped)

    # copy all images from subfolders to a single folder imgs_scraped_clean
    path_imgs_scraped_clean = path_base + "/data/testset/imgs_scraped_clean"
    copy_all_imgs_to_one_folder(
        path_old=path_imgs_scraped,
        path_new=path_imgs_scraped_clean,
    )

    # delete very small images
    delete_small_images(
        path=path_imgs_scraped_clean,
        min_width=85,
        min_height=85,
    )

    # delete images with extreme aspect ratio
    delete_extreme_aspect_ratio_images(
        path=path_imgs_scraped_clean,
        max_aspect_ratio=2.4,
        min_aspect_ratio=0.417,
    )

    # delete duplicates using embeddings and cosine similarity (quite slow)
    delete_duplicate_images(path=path_imgs_scraped_clean, similarity_thesh=0.99999)

    #%% Yolov5 instance segmentation for nail cropping

    path_clone = os.path.join(path_base, "models/yolov5")
    path_yoloScript = path_clone + "/segment/predict.py"
    path_model = path_base + "/models/yolov5_best_model/best.pt"

    # clone yolov5 repo
    if not os.path.exists(path_clone):
        Repo.clone_from("https://github.com/ultralytics/yolov5.git", path_clone)

    path_imgs_scraped_clean_pred = path_base + "/data/testset/imgs_scraped_clean_pred"
    if os.path.exists(path_imgs_scraped_clean_pred):
        shutil.rmtree(path_imgs_scraped_clean_pred)

    # run yolov5 prediction on scraped images
    subprocess.run(
        [
            "python3",
            path_yoloScript,
            "--weights",
            path_model,
            "--source",
            path_imgs_scraped_clean,
            "--save-txt",
            "--name",
            path_imgs_scraped_clean_pred,
            "--project",
            path_imgs_scraped_clean_pred,
            "--conf",
            "0.4",
            "--save-txt",
        ]
    )

    #%% Crop images based on yolov5 prediction, and get new segmentation mask coordinates

    path_imgs_originalSize = os.path.join(path_base, "data/testset/imgs_scraped_clean/")
    path_labels_originalSize = os.path.join(
        path_base, "data/testset/imgs_scraped_clean_pred/labels/"
    )
    path_imgs_cropped = os.path.join(
        path_base, "data/testset/imgs_scraped_clean_pred_cropped/"
    )
    path_labels_cropped = os.path.join(
        path_base, "data/testset/imgs_scraped_clean_pred_cropped/labels"
    )
    labels_originalSize = os.listdir(path_labels_originalSize)

    # go through each original label (txt) file containing predicted segmentation
    # mask coordinates, load the corresponding image file, crop the image based on
    # the mask, get new mask coordinates, and save image and new coordinates.
    for txt_file in labels_originalSize:

        # image file name is the same as txt file name
        img_file = txt_file.split(".")[0] + ".jpg"

        # get number of objects (nails) in the image
        with open(path_labels_originalSize + txt_file, "r") as f:
            nobj_in_image = len(f.readlines())

        # go through each object (nail) in the image
        for cur_obj in range(nobj_in_image):

            # read image and get mask coordinates (polygon_nail)
            image, polygon_nail, obj_class = read_image_label(
                path_imgs_originalSize + img_file,
                path_labels_originalSize + txt_file,
                cur_obj,
            )

            # crop image and get new coordinates
            (
                image_cropped,
                polygon_nail_cropped,
                polygon_nail_cropped_norm,
            ) = crop_image_label(
                image, polygon_nail, square=False, max_extra_pad_prop=0.2, obj_class=0
            )

            # new file names for cropped image and txt file with coordinates:
            # cur_obj at end of file name to distinguish between multiple nails in
            # the same image
            img_file_save = img_file.split(".")[0] + "_" + str(cur_obj) + ".jpg"
            txt_file_save = img_file.split(".")[0] + "_" + str(cur_obj) + ".txt"

            # save cropped image and new txt (label) file with coordinates
            save_image_label(
                image_cropped,
                polygon_nail_cropped_norm,
                path_imgs_cropped,
                path_labels_cropped,
                img_file_save,
                txt_file_save,
            )

    #%% Delete very small crop images and corresponding label (txt) files

    del_imagenames = delete_small_images(
        path=path_imgs_cropped,
        min_width=80,
        min_height=80,
        return_del_filenames=True,
    )

    delete_txt_files_for_del_images(
        file_names_to_delete=del_imagenames,
        path=path_labels_cropped,
    )
