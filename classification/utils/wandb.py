import wandb
import os
from configs.train_config import CFG

def use_wandb():

    try:
        wandb.login(key=os.environ['WANDB_API_KEY'])
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, get your W&B access token from here: https://wandb.ai/authorize'+
             'and save it as an environment variable (WANDB_API_KEY)')

    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    wandb.init(project='Nail project', 
                     name=CFG.experiment_name,
                     config=class2dict(CFG),
                     group=CFG.model_name,
                     job_type="train",
                     anonymous=anony)