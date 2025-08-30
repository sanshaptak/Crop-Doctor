# Placeholder for cnn_models.py
import torch
import torch.nn as nn
import timm

class EfficientNetCropDisease(nn.Module):
    """EfficientNet for Crop Disease Detection"""
    
    def __init__(self, num_classes, model_name='efficientnet_b4', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.backbone(x)

class ResNetCropDisease(nn.Module):
    """ResNet for Crop Disease Detection"""
    
    def __init__(self, num_classes, model_name='resnet50', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
    def forward(self, x):
        return self.backbone(x)

def get_model(model_type, num_classes, model_name=None):
    """Factory function to get different models"""
    
    if model_type == 'vit':
        from .vit_model import ViTCropDisease
        model_name = model_name or 'vit_base_patch16_224'
        return ViTCropDisease(num_classes, model_name)
    
    elif model_type == 'efficientnet':
        model_name = model_name or 'efficientnet_b4'
        return EfficientNetCropDisease(num_classes, model_name)
    
    elif model_type == 'resnet':
        model_name = model_name or 'resnet50'
        return ResNetCropDisease(num_classes, model_name)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
