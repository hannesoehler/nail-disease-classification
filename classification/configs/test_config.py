import os

class CFG_test:
    image_paths = ['imgs_scraped_final']
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'classification/outputs')
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'models/classification')
    num_workers = 0
    seed = 0
    num_classes = 6
    metrics_avg = 'macro'
    wandb = True
    experiment_name = 'best model - testset'
    batch_size_val = 128
    image_size = 256
    model_name = 'resnet152'
