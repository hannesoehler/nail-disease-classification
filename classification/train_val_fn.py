from utils.torch_utils import AverageMeter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils as utils
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, OneCycleLR
import numpy as np
import gc
from configs.train_config import CFG
from configs.test_config import CFG_test
from utils.dataloaders import ImageData
from utils.model import Nail_classifier
from utils.loss import FocalLoss
from utils.metrics import print_loss_and_metrics

def train_fn(model, train_loader, criterion, optimizer, scheduler, device, scaler):
    train_losses = AverageMeter()
    model.train()
    
    for i, (data, target) in enumerate(tqdm(train_loader)): 
        
        target = target.to(device)
        data = data.float().to(device)
        with torch.cuda.amp.autocast(enabled=True):
            output = model(data)
            loss = criterion(output, target)
        train_losses.update(loss.item(), CFG.batch_size_train)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
    return train_losses
             
def valid_fn(model, val_loader, criterion, device):
    val_losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(val_loader)):      
            target = target.to(device)
            data = data.float().to(device)
            
            output = model(data)
                
            loss = criterion(output, target)
                
            val_losses.update(loss.item(), CFG.batch_size_val)
            
            output = F.softmax(output, dim=1)
            
            if CFG.use_folds:
                prob, output = output.max(dim=1)
            else:
                cutoff = 0.9
                output = torch.cat((output[:,:2], output[:,3:5], output[:,2].unsqueeze(1)), dim=1) 
                output_dys = output[:,4] 
                prob, output = output[:,:4].max(dim=1)
                output[output >= 2] = output[output >= 2] + 1
                output[output_dys > cutoff] = 2
                prob[output_dys > cutoff] = output_dys[output_dys > cutoff]
            
            output = output.detach().cpu().numpy()
            prob = prob.detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            if i == 0:
                output_val = output
                prob_val = prob
                target_val = target
            else:
                output_val = np.concatenate((output_val, output))
                prob_val = np.concatenate((prob_val, prob))
                target_val = np.concatenate((target_val, target))
                
    return val_losses, output_val, prob_val, target_val


def train_loop(train_df, fold, device):
    
    train_fold = train_df.loc[train_df['fold']!=fold].reset_index(drop=True)
    class_weights = 1 / (train_fold.label.value_counts() / len(train_fold))
    class_weights = torch.tensor(class_weights).float()
    
    val_fold = train_df.loc[train_df['fold']==fold].reset_index(drop=True)
    
    traindata = ImageData(train_fold, CFG, augment = True)
    
    if CFG.oversampling:
        samples_weight = np.array([class_weights[t] for t in train_fold['label']])
        sampler = utils.sampler.WeightedRandomSampler(samples_weight, len(samples_weight)) 
        train_loader = DataLoader(dataset=traindata,batch_size=CFG.batch_size_train,
                          sampler=sampler, num_workers=CFG.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(dataset=traindata,batch_size=CFG.batch_size_train, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True)
    
    valdata = ImageData(val_fold, CFG)
    val_loader = DataLoader(dataset=valdata,batch_size=CFG.batch_size_val, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=True)
    
    model = Nail_classifier(CFG)
    model = model.to(device)
    
    if CFG.weighted_loss:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif CFG.focal_loss:
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=None)
    
    criterion = criterion.to(device)
    
    if CFG.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=CFG.amsgrad)
    elif CFG.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=CFG.amsgrad)
    elif CFG.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    
    if CFG.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs*len(train_loader), 
                                      eta_min=CFG.min_lr)
    elif CFG.scheduler == 'cosine_with_warmup':
        scheduler = CosineAnnealingWarmRestarts(optimizer, CFG.warm_up, eta_min=CFG.min_lr)
    elif CFG.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, CFG.lr, total_steps=CFG.epochs*len(train_loader))

    f1_lst = []
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    for epoch in range(1, CFG.epochs+1):
            
        train_losses = train_fn(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        
        val_losses, output_val, prob_val, target_val = valid_fn(model, val_loader, criterion, device)
        f1 = print_loss_and_metrics(target_val, output_val, fold=fold, epoch=epoch, train_loss=train_losses, val_loss=val_losses)
        
        f1_lst.append(f1)
        if f1 >= np.max(f1_lst):
            if CFG.use_folds:
                torch.save(model.state_dict(), f'{CFG.model_path}/{CFG.model_name}_fold_{fold}.pt')
            else:
                torch.save(model.state_dict(), f'{CFG.model_path}/{CFG.model_name}.pt')
            val_fold['prediction'] = output_val
            val_fold['probability'] = prob_val
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_fold


def test_loop(val_df, device):
    
    valdata = ImageData(val_df, CFG_test)
    val_loader = DataLoader(dataset=valdata, batch_size=CFG_test.batch_size_val, shuffle=False,
                        num_workers=CFG_test.num_workers, pin_memory=True)
    
    model = Nail_classifier(CFG)
    model = model.to(device)
    model.load_state_dict(torch.load(f'{CFG_test.model_path}/{CFG_test.model_name}.pt', map_location=torch.device(device)))   
    
    criterion = nn.CrossEntropyLoss(weight=None)
    criterion = criterion.to(device)
    
    val_losses, output_val, prob_val, target_val = valid_fn(model, val_loader, criterion, device)
    
    val_df['prediction'] = output_val
    val_df['probability'] = prob_val
               
    torch.cuda.empty_cache()
    gc.collect()
    
    return val_df