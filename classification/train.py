import os
import wandb
import pandas as pd
from utils.torch_utils import seed_torch, use_gpu
from utils.wandb import use_wandb
from configs.train_config import CFG
from utils.data import construct_train_df, create_fold_valid_set, create_kfolds
from train_val_fn import train_loop
from utils.metrics import print_loss_and_metrics


if __name__ == '__main__':
    
    seed_torch(seed=CFG.seed)
    device = use_gpu()
    
    if CFG.wandb:
        use_wandb()

    train_df, class_dict = construct_train_df(CFG.image_paths, train=True)
    
    if CFG.debug:
        train_df = train_df.sample(n=10000).reset_index(drop=True)
    
    if CFG.use_folds:
        train_df = create_kfolds(train_df, CFG.n_fold)
    else:
        train_df = create_fold_valid_set(train_df)
        
    if not os.path.exists(CFG.output_path):
        os.mkdir(CFG.output_path)
    
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(train_df, fold, device)
            oof_df = pd.concat([oof_df, _oof_df])

    _ = print_loss_and_metrics(oof_df['label'], oof_df['prediction'], class_dict, fold=None, epoch=None, train_loss=None, val_loss=None)
    oof_df = oof_df.reset_index(drop=True)
    oof_df.to_csv(CFG.output_path+'/oof_df.csv')
        
    if CFG.wandb:
        wandb.finish()