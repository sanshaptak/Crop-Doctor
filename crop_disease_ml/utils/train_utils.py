# Placeholder for train_utils.py
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import copy

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    
    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        # Update progress bar
        current_loss = running_loss / total_samples
        current_acc = running_corrects.double() / total_samples * 100
        
        progress_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples * 100
    
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch"""
    
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} - Validation")
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # Update progress bar
            current_loss = running_loss / total_samples
            current_acc = running_corrects.double() / total_samples * 100
            
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples * 100
    
    return epoch_loss, epoch_acc.item()

def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, val_loss, class_names, filepath, is_best=False):
    """Save model checkpoint"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'class_names': class_names,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        print(f"✅ New best model saved! Accuracy: {val_acc:.2f}%")
    else:
        print(f"💾 Checkpoint saved at epoch {epoch}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cuda'):
    """Load model checkpoint"""
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

class EarlyStopping:
    """Early stopping to avoid overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = copy.deepcopy(model.state_dict())
