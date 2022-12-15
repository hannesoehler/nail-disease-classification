import os

class CFG:
    image_paths = ['dataset (A1) - single nails', 'dataset (A2) - single nails',
                 'dataset B1', 'dataset B2', 'dataset C', 'dataset D', 'Virtual E Dataset']
    train_with_e_dataset = True
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'classification/outputs')
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'models/classification')
    num_workers = 0
    seed = 0
    num_classes = 6
    use_folds = False
    n_fold = 5
    trn_fold = [0]
    metrics_avg = 'macro'
    wandb = True
    experiment_name = 'same valset distribution as testset'
    debug = False
    
    model_name = 'resnet152'
    epochs = 10
    lr = 5e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    batch_size_train = 64
    batch_size_val = 128
    image_size = 256
    optimizer = 'Adam'
    amsgrad = True
    scheduler = 'cosine'
    warm_up = 0
    oversampling = False
    weighted_loss = False
    focal_loss = False
    aug_prob = 0.5
    mixup = True
    mixup_alpha = 5
    mixup_beta = 5