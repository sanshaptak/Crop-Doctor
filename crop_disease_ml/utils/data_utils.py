# Placeholder for data_utils.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import os
import pandas as pd

class CropDiseaseDataset(Dataset):
    """Custom dataset for crop disease images"""
    
    def __init__(self, root_dir, transform=None, csv_file=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if csv_file and os.path.exists(csv_file):
            # Load from CSV file
            self.df = pd.read_csv(csv_file)
            self.samples = [(row['image_path'], row['label']) for _, row in self.df.iterrows()]
            self.classes = sorted(self.df['class_name'].unique().tolist())
        else:
            # Use ImageFolder structure
            self.dataset = datasets.ImageFolder(root_dir)
            self.samples = self.dataset.samples
            self.classes = self.dataset.classes
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if hasattr(self, 'dataset'):
            image, label = self.dataset[idx]
        else:
            image_path, label = self.samples[idx]
            image = Image.open(os.path.join(self.root_dir, image_path)).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(image_size=224, augment=True):
    """Get data transforms"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_dataloaders(train_dir, val_dir, batch_size=16, num_workers=4, image_size=224):
    """Create data loaders"""
    
    train_transform, val_transform = get_transforms(image_size)
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, train_dataset.classes

def check_data_structure(data_dir):
    """Check if data directory has correct structure"""
    
    if not os.path.exists(data_dir):
        return False, f"Directory {data_dir} not found"
    
    subdirs = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    if len(subdirs) == 0:
        return False, f"No class directories found in {data_dir}"
    
    class_info = {}
    total_images = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        images = [f for f in os.listdir(subdir_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        class_info[subdir] = len(images)
        total_images += len(images)
    
    print(f"\nDataset Analysis for {data_dir}:")
    print("-" * 40)
    for class_name, count in sorted(class_info.items()):
        print(f"{class_name}: {count} images")
    print("-" * 40)
    print(f"Total: {len(subdirs)} classes, {total_images} images")
    
    return True, f"Found {len(subdirs)} classes with {total_images} total images"

def create_sample_data_structure():
    """Create sample data structure for testing"""
    
    base_dirs = ['data/train', 'data/val', 'data/test', 'data/sample']
    classes = ['healthy_apple', 'apple_scab', 'healthy_tomato', 'tomato_blight', 
               'healthy_rice', 'rice_blast']
    
    for base_dir in base_dirs:
        if base_dir == 'data/sample':
            os.makedirs(base_dir, exist_ok=True)
            continue
            
        for class_name in classes:
            class_dir = os.path.join(base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    print("Sample data structure created!")
    print("Please add your images to the appropriate directories.")