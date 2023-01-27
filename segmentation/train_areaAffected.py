import os
import subprocess
from git import Repo

#!pip install roboflow
#!pip install wandb
from roboflow import Roboflow

if __name__ == "__main__":

    path_base = os.path.dirname(os.path.dirname(__file__))
    path_clone = os.path.join(path_base, "models/yolov5")

    # clone yolov5 repo
    if not os.path.exists(path_clone):
        Repo.clone_from("https://github.com/ultralytics/yolov5.git", path_clone)
    os.chdir(path_clone)

    # download dataset with area-affected annotations
    rf = Roboflow(api_key="INSERTYOURAPIKEYHERE")
    project = rf.workspace("tim-lauer-k1awl").project("onychomycosis-segmentation-v4")
    dataset = project.version(2).download("yolov5")

    # train area-affected model
    path_yoloScript = path_clone + "/segment/train.py"
    path_data = path_clone + "/Onychomycosis-segmentation-v4-2/data.yaml"
    pretrained_model = "yolov5x-seg.pt"
    name_project = "severity_seg_v2"
    name_run = "yolov5x-seg_results_area_affected_final"
    batch_size = "32"
    n_epochs = "700"
    save_period = "2"
    patience_epochs_when_no_improvement = "200"

    subprocess.run(
        [
            "python3",
            path_yoloScript,
            "--batch",
            batch_size,
            "--epochs",
            n_epochs,
            "--data",
            path_data,
            "--weights",
            pretrained_model,
            "--project",
            name_project,
            "--name",
            name_run,
            "--save-period",
            save_period,
            "--patience",
            patience_epochs_when_no_improvement,
            "--cache",
        ]
    )
