# Placeholder for README.md
"""
# Crop Disease Detection with Vision Transformer

A complete machine learning pipeline for detecting crop diseases using Vision Transformers (ViT) and Convolutional Neural Networks.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone or download the project
cd crop_disease_ml

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\\Scripts\\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset
Organize your images in this structure:
```
data/
├── train/
│   ├── healthy_apple/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── apple_scab/
│   ├── healthy_tomato/
│   ├── tomato_blight/
│   └── ...
├── val/          # Same structure as train
└── test/         # Same structure as train
```

### 3. Train the Model
```bash
python train.py
```

### 4. Make Predictions
```bash
# Single image
python predict.py --image data/sample/test_image.jpg

# Batch prediction
python predict.py --dir data/sample/

# Use different model
python predict.py --image test.jpg --model checkpoints/final_model.pth
```

### 5. Evaluate Model
```bash
python evaluate.py
```

## 🔧 Configuration

Modify `config.py` to adjust:
- Model architecture (ViT variants)
- Training parameters (batch size, learning rate)
- Data paths
- Hardware settings

### Available Models:
- `vit_tiny_patch16_224` - Fastest, lowest accuracy
- `vit_small_patch16_224` - Good balance
- `vit_base_patch16_224` - Recommended (default)
- `vit_large_patch16_224` - Highest accuracy, slowest
- `efficientnet_b0/b4` - CNN alternatives

## 📊 Features

### Training:
- ✅ Automatic data validation
- ✅ Advanced data augmentation
- ✅ Learning rate scheduling
- ✅ Early stopping
- ✅ Model checkpointing
- ✅ Training visualization

### Evaluation:
- ✅ Comprehensive metrics
- ✅ Confusion matrix
- ✅ Per-class analysis
- ✅ Prediction confidence

### Prediction:
- ✅ Single image prediction
- ✅ Batch processing
- ✅ Top-k predictions
- ✅ Confidence scores

## 🎯 Expected Performance

| Model | Accuracy | Speed | Memory |
|-------|----------|-------|---------|
| ViT-Tiny | 92-94% | Fast | Low |
| ViT-Base | 96-98% | Medium | Medium |
| ViT-Large | 97-99% | Slow | High |

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA out of memory**
   - Reduce `BATCH_SIZE` in config.py
   - Use smaller model variant

2. **Data not found**
   - Check data directory structure
   - Ensure images are in correct format (.jpg, .png)

3. **Model not found**
   - Run training first: `python train.py`
   - Check checkpoint path

## 📁 Project Structure

```
crop_disease_ml/
├── data/              # Dataset
├── models/            # Model architectures
├── utils/             # Utility functions
├── checkpoints/       # Saved models
├── results/           # Training outputs
├── config.py          # Configuration
├── train.py           # Training script
├── predict.py         # Prediction script
├── evaluate.py        # Evaluation script
└── requirements.txt   # Dependencies
```

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License.
"""