import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
import numpy as np

from config import Config
from models.vit_model import ViTCropDisease

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Initialize model
    model = ViTCropDisease(num_classes, Config.MODEL_NAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    print(f"✅ Model loaded successfully!")
    print(f"📋 Classes: {class_names}")
    print(f"🎯 Best accuracy: {checkpoint.get('val_acc', 'Unknown'):.2f}%")
    
    return model, class_names

def predict_single_image(model, image_path, class_names, device, top_k=3):
    """Predict disease for a single image"""
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, len(class_names)))
        
        # Convert to lists
        top_k_probs = top_k_probs.cpu().numpy()[0]
        top_k_indices = top_k_indices.cpu().numpy()[0]
    
    # Prepare results
    predictions = []
    for i in range(len(top_k_indices)):
        predictions.append({
            'class': class_names[top_k_indices[i]],
            'confidence': float(top_k_probs[i]),
            'probability': float(top_k_probs[i] * 100)
        })
    
    return {
        'image_path': image_path,
        'predictions': predictions,
        'top_prediction': predictions[0]
    }

def predict_batch(model, image_dir, class_names, device):
    """Predict for all images in a directory"""
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"🔍 Found {len(image_files)} images to predict")
    
    results = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        try:
            result = predict_single_image(model, image_path, class_names, device)
            results.append(result)
            
            # Print result
            top_pred = result['top_prediction']
            print(f"📷 {image_file}")
            print(f"   🎯 {top_pred['class']} ({top_pred['probability']:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error predicting {image_file}: {e}")
    
    return results

def main():
    """Main prediction function"""
    
    parser = argparse.ArgumentParser(description='Predict crop disease from images')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory containing images')
    parser.add_argument('--model', type=str, default=Config.BEST_MODEL_PATH, 
                       help='Path to model checkpoint')
    parser.add_argument('--top-k', type=int, default=3, 
                       help='Show top-k predictions')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.dir:
        print("❌ Please provide either --image or --dir argument")
        return
    
    if args.image and args.dir:
        print("❌ Please provide only one of --image or --dir, not both")
        return
    
    # Load model
    try:
        model, class_names = load_model(args.model, Config.DEVICE)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\n🔮 Making predictions...")
    print("=" * 50)
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
            
        try:
            result = predict_single_image(model, args.image, class_names, Config.DEVICE, args.top_k)
            
            print(f"📷 Image: {result['image_path']}")
            print(f"🎯 Top Prediction: {result['top_prediction']['class']}")
            print(f"🎲 Confidence: {result['top_prediction']['probability']:.2f}%")
            
            if len(result['predictions']) > 1:
                print(f"\n📊 Top {args.top_k} Predictions:")
                for i, pred in enumerate(result['predictions'], 1):
                    print(f"   {i}. {pred['class']}: {pred['probability']:.2f}%")
                    
        except Exception as e:
            print(f"❌ Prediction error: {e}")
    
    # Directory batch prediction
    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"❌ Directory not found: {args.dir}")
            return
            
        results = predict_batch(model, args.dir, class_names, Config.DEVICE)
        
        if results:
            print(f"\n📊 Batch Prediction Summary:")
            print(f"   Total images: {len(results)}")
            
            # Count predictions by class
            class_counts = {}
            for result in results:
                pred_class = result['top_prediction']['class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
            
            print(f"   Predictions by class:")
            for class_name, count in sorted(class_counts.items()):
                print(f"     {class_name}: {count}")

if __name__ == "__main__":
    main()
