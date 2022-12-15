import albumentations
import cv2
import numpy as np
from torch.utils.data import Dataset
from configs.train_config import CFG


class Transform:
    def __init__(self):
        self.transform = albumentations.Compose([
   albumentations.RandomResizedCrop(CFG.image_size, CFG.image_size, scale=(0.8, 1), p=CFG.aug_prob), 
   albumentations.HorizontalFlip(p=CFG.aug_prob),
   albumentations.VerticalFlip(p=CFG.aug_prob),
   albumentations.ShiftScaleRotate(p=CFG.aug_prob),
   albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=CFG.aug_prob),
   albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=CFG.aug_prob),
   albumentations.CLAHE(clip_limit=(1,4), p=CFG.aug_prob),
   albumentations.ImageCompression (quality_lower=80, quality_upper=100, p=CFG.aug_prob),
   albumentations.Resize(CFG.image_size, CFG.image_size),
   albumentations.Cutout(max_h_size=int(CFG.image_size * 0.1), max_w_size=int(CFG.image_size * 0.1), num_holes=5, p=CFG.aug_prob)
        ])
        
    def __call__(self, image):
        image = self.transform(image=image)["image"]
        return image
 
    
class Valid_Transform:
    def __init__(self):
        self.transform = albumentations.Compose([
  albumentations.Resize(CFG.image_size, CFG.image_size)
        ])
        
    def __call__(self, image):
        image = self.transform(image=image)["image"]
        return image
    

class ImageData(Dataset):
    def __init__(self, labels_df, CFG, augment=False):
        super().__init__()
        self.cfg = CFG
        self.labels_df = labels_df
        self.augmentation = Transform()
        self.valid_transform = Valid_Transform()
        self.augment = augment
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, index):       
        label = self.labels_df.label[index]
        image_path = self.labels_df.image_path[index]
        image = cv2.imread(image_path)

        if self.augment and self.cfg.mixup and np.random.random() > self.cfg.aug_prob:
            i = np.random.randint(len(self.labels_df))
            while self.labels_df.label[i] != label:
                i = np.random.randint(len(self.labels_df))
            image_path = self.labels_df.image_path[i]
            mixup_image = cv2.imread(image_path)
            image = cv2.resize(image, (self.cfg.image_size, self.cfg.image_size), interpolation= cv2.INTER_LINEAR)
            mixup_image = cv2.resize(mixup_image, (self.cfg.image_size, self.cfg.image_size), interpolation= cv2.INTER_LINEAR)
            mixup_ratio = np.random.beta(self.cfg.mixup_alpha, self.cfg.mixup_beta)
            image = np.round(image * mixup_ratio + mixup_image * (1 - mixup_ratio)).astype(np.uint8)
           
        if self.augment:
            image = self.augmentation(image)
        else:
            image = self.valid_transform(image)
            
        image = image.transpose((2,0,1))  
        return image, label
    
