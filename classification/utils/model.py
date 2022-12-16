import torch.nn as nn
import timm
from configs.train_config import CFG


class Nail_classifier(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.cfg = CFG
        self.model = timm.create_model(self.cfg.model_name, pretrained=True, in_chans=3)
        if 'efficientnet' in self.cfg.model_name:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif 'resnet' in self.cfg.model_name:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif 'swin' in self.cfg.model_name:
            self.n_features = self.model.head.in_features
            self.model.head = nn.Identity()
        self.linear = nn.Linear(self.n_features, self.cfg.num_classes)
        
    def forward(self, x):
        last_layer = self.model(x)
        output = self.linear(last_layer)
        return output