# Placeholder for eval_utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm

def evaluate_model(model, test_loader, class_names, device):
    """Comprehensive model evaluation"""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"{'='*50}")
    
    # Detailed classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nDetailed Classification Report:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report
    }

def plot_confusion_matrix(true_labels, predictions, class_names, save_path=None):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(12, 10))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annotations = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_training_history(history, save_path=None):
    """Plot training history"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    if 'learning_rate' in history:
        ax3.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')
    
    # Best accuracy indicator
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    
    ax4.text(0.1, 0.8, f'Best Validation Accuracy:', fontsize=12, fontweight='bold')
    ax4.text(0.1, 0.6, f'{best_val_acc:.2f}%', fontsize=20, fontweight='bold', color='red')
    ax4.text(0.1, 0.4, f'Achieved at Epoch: {best_epoch}', fontsize=12)
    ax4.text(0.1, 0.2, f'Total Training Time: {history.get("training_time", "N/A")}', fontsize=10)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()

def analyze_predictions(true_labels, predictions, probabilities, class_names):
    """Analyze model predictions"""
    
    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        mask = (true_labels == i)
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == i)
            per_class_acc[class_name] = acc
    
    # Sort by accuracy
    sorted_acc = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    for class_name, acc in sorted_acc:
        print(f"{class_name}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Find most confused classes
    cm = confusion_matrix(true_labels, predictions)
    
    print("\nMost Confused Class Pairs:")
    print("-" * 40)
    
    max_confusion = 0
    most_confused = None
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > max_confusion:
                max_confusion = cm[i, j]
                most_confused = (class_names[i], class_names[j])
    
    if most_confused:
        print(f"Most confused: {most_confused[0]} → {most_confused[1]} ({max_confusion} cases)")
    
    return per_class_acc
