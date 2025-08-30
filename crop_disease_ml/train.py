import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

from config import Config
from models.vit_model import ViTCropDisease
from utils.data_utils import get_dataloaders, check_data_structure
from utils.train_utils import train_epoch, validate_epoch, save_checkpoint, EarlyStopping
from utils.eval_utils import plot_training_history

def main():
    """Main training function"""
    
    # Setup
    Config.create_dirs()
    Config.print_config()
    
    print(f"🚀 Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check data structure
    print("\n📁 Checking data structure...")
    for split in ['train', 'val']:
        data_dir = getattr(Config, f"{split.upper()}_DIR")
        is_valid, message = check_data_structure(data_dir)
        if not is_valid:
            print(f"❌ Error in {split} data: {message}")
            print("\n💡 To create sample structure, run:")
            print("from utils.data_utils import create_sample_data_structure")
            print("create_sample_data_structure()")
            return
        print(f"✅ {split.capitalize()} data: {message}")
    
    # Create data loaders
    print("\n📊 Loading datasets...")
    try:
        train_loader, val_loader, class_names = get_dataloaders(
            Config.TRAIN_DIR, 
            Config.VAL_DIR, 
            Config.BATCH_SIZE, 
            Config.NUM_WORKERS, 
            Config.IMAGE_SIZE
        )
        
        num_classes = len(class_names)
        print(f"✅ Loaded datasets successfully!")
        print(f"📋 Found {num_classes} classes: {class_names}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize model
    print(f"\n🤖 Initializing {Config.MODEL_NAME} model...")
    model = ViTCropDisease(num_classes, Config.MODEL_NAME).to(Config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': Config.LEARNING_RATE * 0.1},  # Lower LR for pre-trained
        {'params': model.backbone.head.parameters(), 'lr': Config.LEARNING_RATE}    # Higher LR for new head
    ], weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.LEARNING_RATE * 0.01
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    print(f"\n🎯 Starting training for {Config.NUM_EPOCHS} epochs...")
    print("=" * 70)
    
    # Training loop
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print(f"\n📅 Epoch {epoch}/{Config.NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, epoch
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, Config.DEVICE, epoch
        )
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        # Print epoch results
        print(f"📈 Results:")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"   LR:    {current_lr:.2e}")
        
        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_acc, val_loss, 
            class_names, Config.BEST_MODEL_PATH, is_best
        )
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"🛑 Early stopping triggered at epoch {epoch}")
            break
    
    # Training completed
    training_time = time.time() - start_time
    history['training_time'] = f"{training_time/60:.1f} minutes"
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {training_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    print(f"\n📊 Generating training plots...")
    plot_training_history(history, os.path.join(Config.PLOTS_DIR, "training_history.png"))
    
    # Save final model
    final_model_path = os.path.join(Config.CHECKPOINT_DIR, "final_model.pth")
    save_checkpoint(
        model, optimizer, scheduler, epoch, val_acc, val_loss, 
        class_names, final_model_path
    )
    
    print(f"💾 Models saved:")
    print(f"   Best: {Config.BEST_MODEL_PATH}")
    print(f"   Final: {final_model_path}")
    
    return history

if __name__ == "__main__":
    main()
