import os
import shutil
import subprocess
from git.repo.base import Repo

if __name__ == "__main__":

    path_base = os.path.dirname(os.path.dirname(__file__))
    path_clone = os.path.join(path_base, "models/yolov5")

    # clone yolov5 repo
    if not os.path.exists(path_clone):
        Repo.clone_from("https://github.com/ultralytics/yolov5.git", path_clone)
    os.chdir(path_clone)

    # whole nail model prediction
    path_yoloScript = path_clone + "/segment/predict.py"
    model = (
        path_base
        + "/models/yolov5/severity_seg_v2/yolov5x-seg_results_area_affected_final/weights/best.pt"
    )
    path_imgs_scraped_clean = (
        path_base + "/data/valset/segmentation/raw/imgs_scraped_clean"
    )
    path_imgs_scraped_clean_pred_areaAffected = (
        path_base + "/data/valset/segmentation/raw/imgs_scraped_clean_pred_areaAffected"
    )
    if os.path.exists(path_imgs_scraped_clean_pred_areaAffected):
        shutil.rmtree(path_imgs_scraped_clean_pred_areaAffected)

    conf_thres = "0.067"
    iou_thresh = "0.6"
    line_thick = "1"

    subprocess.run(
        [
            "python3",
            path_yoloScript,
            "--weights",
            model,
            "--line-thickness",
            line_thick,
            "--source",
            path_imgs_scraped_clean,
            "--name",
            path_imgs_scraped_clean_pred_areaAffected,
            "--project",
            path_imgs_scraped_clean_pred_areaAffected,
            "--iou-thres",
            iou_thresh,
            "--conf-thres",
            conf_thres,
            "--save-txt",
        ]
    )
