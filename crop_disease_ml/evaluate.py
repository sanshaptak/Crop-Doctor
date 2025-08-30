# Placeholder for evaluate.py
import torch
import os
from config import Config
from models.vit_model import ViTCropDisease
from utils.data_utils import get_dataloaders, check_data_structure
from utils.eval_utils import evaluate_model, plot_confusion_matrix, analyze_predictions
from utils.train_utils import load_checkpoint

def main():
    """Main evaluation function"""
    
    print("🔍 Model Evaluation")
    print("=" * 50)
    
    # Check if test data exists
    if not os.path.exists(Config.TEST_DIR):
        print(f"❌ Test directory not found: {Config.TEST_DIR}")
        print("💡 Using validation data for evaluation...")
        test_dir = Config.VAL_DIR
    else:
        test_dir = Config.TEST_DIR
    
    # Check data structure
    is_valid, message = check_data_structure(test_dir)
    if not is_valid:
        print(f"❌ Error: {message}")
        return
    print(f"✅ Test data: {message}")
    
    # Load test data
    print("\n📊 Loading test dataset...")
    try:
        # We use train_dir to get class_names, but test_dir for actual testing
        _, test_loader, class_names = get_dataloaders(
            Config.TRAIN_DIR, test_dir, 
            Config.BATCH_SIZE, Config.NUM_WORKERS, Config.IMAGE_SIZE
        )
        print(f"✅ Test data loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Load trained model
    print(f"\n🤖 Loading trained model...")
    if not os.path.exists(Config.BEST_MODEL_PATH):
        print(f"❌ Model checkpoint not found: {Config.BEST_MODEL_PATH}")
        print("💡 Please train the model first using: python train.py")
        return
    
    try:
        checkpoint = load_checkpoint(Config.BEST_MODEL_PATH, None, device=Config.DEVICE)
        
        num_classes = len(checkpoint['class_names'])
        model = ViTCropDisease(num_classes, Config.MODEL_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(Config.DEVICE)
        
        print(f"✅ Model loaded successfully!")
        print(f"🎯 Training accuracy: {checkpoint.get('val_acc', 'Unknown'):.2f}%")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Evaluate model
    print(f"\n🧪 Running evaluation...")
    results = evaluate_model(model, test_loader, class_names, Config.DEVICE)
    
    # Plot confusion matrix
    print(f"\n📊 Generating confusion matrix...")
    confusion_matrix_path = os.path.join(Config.PLOTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(
        results['true_labels'], 
        results['predictions'], 
        class_names, 
        confusion_matrix_path
    )
    
    # Analyze predictions
    print(f"\n🔬 Analyzing predictions...")
    per_class_acc = analyze_predictions(
        results['true_labels'], 
        results['predictions'], 
        results['probabilities'], 
        class_names
    )
    
    # Save results to file
    results_file = os.path.join(Config.RESULTS_DIR, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("CROP DISEASE DETECTION - EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {Config.MODEL_NAME}\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")
        
        f.write("PER-CLASS ACCURACY:\n")
        f.write("-" * 30 + "\n")
        for class_name, acc in sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{class_name}: {acc:.4f} ({acc*100:.2f}%)\n")
        
        f.write(f"\nDETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 40 + "\n")
        f.write(results['classification_report'])
    
    print(f"📄 Detailed results saved to: {results_file}")
    
    # Summary
    print(f"\n🎉 Evaluation Summary:")
    print(f"   📊 Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"   🏆 Best Class: {max(per_class_acc.items(), key=lambda x: x[1])[0]}")
    print(f"   📉 Worst Class: {min(per_class_acc.items(), key=lambda x: x[1])[0]}")
    print(f"   📁 Results saved in: {Config.RESULTS_DIR}")

if __name__ == "__main__":
    main()