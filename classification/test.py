import os
import wandb
from utils.torch_utils import seed_torch, use_gpu
from utils.wandb import use_wandb
from configs.test_config import CFG_test
from train_val_fn import test_loop
from utils.metrics import print_loss_and_metrics
from utils.data import construct_train_df


if __name__ == '__main__':
    seed_torch(seed=CFG_test.seed)
    device = use_gpu()
    
    if CFG_test.wandb:
        use_wandb()
        
    test_df, class_dict = construct_train_df(CFG_test.image_paths, train=False)
    
    if not os.path.exists(CFG_test.output_path):
        os.mkdir(CFG_test.output_path)

    oof_df = test_loop(test_df, device)
    _ = print_loss_and_metrics(oof_df['label'], oof_df['prediction'], class_dict, fold=None, epoch=None, train_loss=None, val_loss=None)

    if CFG_test.wandb:
        wandb.finish()