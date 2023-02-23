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

    # download dataset with whole-nail annotations
    rf = Roboflow(api_key="INSERTYOURAPIKEYHERE")
    project = rf.workspace("tim-lauer-k1awl").project("nail-segmentation")
    dataset = project.version(2).download("yolov5")

    # train whole-nail model
    path_yoloScript = path_clone + "/segment/train.py"
    path_data = path_clone + "/Nail-segmentation-2/data.yaml"
    pretrained_model = "yolov5x-seg.pt"
    name_project = "nail_seg_v2"
    name_run = "yolov5x-nailseg_results_final"
    batch_size = "32"
    n_epochs = "200"
    save_period = "5"
    patience_epochs_when_no_improvement = "100"

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
