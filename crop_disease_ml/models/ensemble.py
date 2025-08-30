# Placeholder for ensemble.py
import torch
import torch.nn as nn
from .vit_model import ViTCropDisease
from .cnn_models import EfficientNetCropDisease

class EnsembleModel(nn.Module):
    """Ensemble of different models"""
    
    def __init__(self, num_classes, models_config):
        super().__init__()
        self.models = nn.ModuleList()
        self.weights = []
        
        for config in models_config:
            if config['type'] == 'vit':
                model = ViTCropDisease(num_classes, config['name'])
            elif config['type'] == 'efficientnet':
                model = EfficientNetCropDisease(num_classes, config['name'])
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
                
            self.models.append(model)
            self.weights.append(config.get('weight', 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average
        weighted_output = sum(w * out for w, out in zip(self.weights, outputs))
        return weighted_output