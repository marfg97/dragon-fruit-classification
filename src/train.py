import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from models import SimpleCNN, torch_models
import json
from config import TrainingConfig

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_transforms(augmentation_intensity=0.8, is_train=True):
    """Transformaciones con aumento de datos solo para entrenamiento"""
    base = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    if is_train and augmentation_intensity > 0:
        base = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.2*augmentation_intensity,
                contrast=0.2*augmentation_intensity,
                saturation=0.2*augmentation_intensity
            )
        ] + base
    
    return transforms.Compose(base)

def get_class_distribution(dataset):
    """Calcula distribución de clases"""
    counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        counts[label] += 1
    return counts

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': running_loss / len(dataloader),
        'accuracy': np.mean(np.array(all_preds) == np.array(all_labels)),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }
    
    # Métricas por clase
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0)
    
    for i, class_name in enumerate(dataloader.dataset.dataset.classes):
        metrics.update({
            f'precision_{class_name}': precision[i],
            f'recall_{class_name}': recall[i],
            f'f1_{class_name}': f1[i]
        })
    
    return metrics



def main():
    try:
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Get configuration
        args = TrainingConfig.get_args()
        logger.info("Training configuration loaded")
        logger.info(f"\n{json.dumps(vars(args), indent=4, sort_keys=True)}")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        
        # Create model directory
        os.makedirs(args.model_dir, exist_ok=True)
        logger.info(f"Model directory: {args.model_dir}")
        
        # Load and transform data
        train_transforms = create_transforms(args.augmentation_intensity, is_train=True)
        test_transforms = create_transforms(0.0, is_train=False)
        
        full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transforms)
        logger.info(f"Dataset loaded from {args.data_dir}")
        logger.info(f"Classes: {full_dataset.classes}")
        logger.info(f"Class counts: {get_class_distribution(full_dataset).tolist()}")
        
        # Convert class weights
        class_weights = torch.tensor(json.loads(args.class_weights), dtype=torch.float32)
        logger.info(f"Class weights: {class_weights.tolist()}")
        
        # Split dataset
        val_size = int(args.val_split * len(full_dataset))
        test_size = int(args.test_size * len(full_dataset))
        train_size = len(full_dataset) - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        test_dataset.dataset.transform = test_transforms
        
        # Create sampler if enabled
        train_sampler = None
        if args.use_sampler:
            sample_weights = class_weights[torch.tensor(
                [label for _, label in full_dataset.samples]
            )][train_dataset.indices]
            train_sampler = WeightedRandomSampler(
                sample_weights, 
                len(train_dataset), 
                replacement=True
            )
            logger.info("Using weighted random sampler")
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        num_classes=len(full_dataset.classes)
        # Initialize model
        # model = SimpleCNN(num_classes,dropout_rate=args.dropout_rate)
        model = torch_models.get_resnet18(num_classes)
        model.to(device)
        logger.info(f"Model initialized:\n{model}")

        
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            patience=3, 
            factor=0.5
        )
        
        # Training loop
        best_metric = -float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{args.epochs} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Avg Loss: {train_loss/(batch_idx+1):.4f}"
                    )
            
            # Validation
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            # Check for early stopping
            current_metric = val_metrics.get(args.monitor_metric, val_metrics['accuracy'])
            if current_metric > best_metric + args.min_delta:
                best_metric = current_metric
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'class_to_idx': full_dataset.class_to_idx,
                    'args': vars(args)
                }, os.path.join(args.model_dir, 'best_model.pth'))
                logger.info(f"New best model saved with {args.monitor_metric}: {best_metric:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience and args.early_stopping:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Update learning rate
            scheduler.step(val_metrics['accuracy'])
            
            # Log epoch results
            logger.info(f"\nEpoch {epoch+1} Summary:")
            logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            for class_name in full_dataset.classes:
                logger.info(
                    f"{class_name} - "
                    f"Precision: {val_metrics[f'precision_{class_name}']:.4f} | "
                    f"Recall: {val_metrics[f'recall_{class_name}']:.4f} | "
                    f"F1: {val_metrics[f'f1_{class_name}']:.4f}"
                )
        
        # Final evaluation
        logger.info("\nTraining completed. Evaluating on test set...")
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        logger.info("\nTest Results:")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        for class_name in full_dataset.classes:
            logger.info(
                f"{class_name} - "
                f"Precision: {test_metrics[f'precision_{class_name}']:.4f} | "
                f"Recall: {test_metrics[f'recall_{class_name}']:.4f} | "
                f"F1: {test_metrics[f'f1_{class_name}']:.4f}"
            )
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': full_dataset.class_to_idx,
            'test_metrics': test_metrics,
            'args': vars(args)
        }, os.path.join(args.model_dir, 'final_model.pth'))
        logger.info(f"Final model saved to {args.model_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()