import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from configs.train_config import CFG


def construct_train_df(image_paths, train=True):
    
    if train:
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'data/trainsets/')
    else:
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),'data/testset/')
    images_lst = []
    class_dict = {}
    
    for path in image_paths:
        path = base_path+path
        for image_folder in glob(path+'/*/'):
            for image_path in glob(image_folder+'/*'):
                if 'onychomycosis' in image_path.lower():
                    label = 'onychomycosis'
                    label_idx = 1
                    class_dict[label_idx] = label
                elif 'naildystrophy' in image_path.lower():
                    label = 'nail dystrophy'
                    label_idx = 2
                    class_dict[label_idx] = label
                elif 'onycholysis'in image_path.lower():
                    label = 'onycholysis'
                    label_idx = 3
                    class_dict[label_idx] = label
                elif 'melanonychia' in image_path.lower():
                    label = 'melanonychia'
                    label_idx = 4
                    class_dict[label_idx] = label
                elif any(disease in image_path.lower() for disease in ['etc', 'pincer', 'whitespot', 'nodule', 'others']):  
                    # not used: 'atypical', 'focus', 'undetermined'
                    label = 'other disease'
                    label_idx = 5
                    class_dict[label_idx] = label
                elif 'normal' in image_path.lower():
                    label = 'normal nail'
                    label_idx = 0
                    class_dict[label_idx] = label
                else:
                    continue
                image = [image_path, label,  label_idx, image_folder]
                images_lst.append(image)
            
    train_df = pd.DataFrame(images_lst, columns=['image_path', 'disease', 'label', 'image_folder'])
    CFG.class_dict = class_dict
    print(train_df.disease.value_counts())
    return train_df, class_dict


def create_fold_valid_set(df):
    class_freq = [224, 200, 127, 237, 181]
    val_set = pd.DataFrame()
    for i in range(CFG.num_classes-1):
        mask_class = df.apply(lambda row: 'Virtual E Dataset' in row['image_folder'] and row['label']==i, axis=1)
        class_df = df[mask_class].sample(n=min(class_freq[i], np.sum(mask_class)))
        val_set = val_set.append(class_df)
    val_set['fold'] = 0
    df = df.join(val_set[['fold']])
    if CFG.train_with_e_dataset == False:
        df = df[df.apply(lambda row: 'Virtual E Dataset' not in row['image_folder'] or row['fold']==0, axis=1)]
    return df


def create_kfolds(df, n_fold):
    gkf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=CFG.seed)

    df['fold'] = -1
    for fold, (train, val) in enumerate(gkf.split(df['image_path'], df['label'], df['image_folder'])): 
        df.loc[val,'fold']=fold
    print(df.groupby('fold').disease.value_counts())
    return df