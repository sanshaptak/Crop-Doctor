import torch
import os

class Config:
    # Paths
    DATA_ROOT = "data"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    SAMPLE_DIR = os.path.join(DATA_ROOT, "sample")
    
    # Model settings
    MODEL_NAME = 'vit_base_patch16_224'  # Options: vit_tiny, vit_small, vit_base, vit_large
    IMAGE_SIZE = 224
    NUM_CLASSES = None  # Will be set automatically based on dataset
    
    # Training settings
    BATCH_SIZE = 16  # Reduce to 8 if GPU memory issues
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4 if torch.cuda.is_available() else 2
    
    # Checkpoints and results
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    LOGS_DIR = os.path.join(CHECKPOINT_DIR, "training_logs")
    
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    # Model variants available
    MODEL_VARIANTS = {
        'vit_tiny_patch16_224': 'ViT-Tiny (5M params) - Fastest',
        'vit_small_patch16_224': 'ViT-Small (22M params) - Good balance',
        'vit_base_patch16_224': 'ViT-Base (86M params) - Recommended',
        'vit_large_patch16_224': 'ViT-Large (307M params) - Highest accuracy',
        'efficientnet_b0': 'EfficientNet-B0 - CNN alternative',
        'efficientnet_b4': 'EfficientNet-B4 - High accuracy CNN'
    }
    
    @staticmethod
    def create_dirs():
        """Create necessary directories"""
        dirs = [Config.CHECKPOINT_DIR, Config.RESULTS_DIR, Config.PLOTS_DIR, Config.LOGS_DIR]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
        print("Created necessary directories")

    @staticmethod
    def print_config():
        """Print current configuration"""
        print("=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        print(f"Device: {Config.DEVICE}")
        print(f"Model: {Config.MODEL_NAME}")
        print(f"Image Size: {Config.IMAGE_SIZE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Epochs: {Config.NUM_EPOCHS}")
        print("=" * 50)