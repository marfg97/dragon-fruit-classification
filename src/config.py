import os
import argparse

class TrainingConfig:
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description='Train a CNN model for defect classification')
        
        # Hyperparameters
        parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=48, help='Batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.0003, help='Initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay for optimizer')
        parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate for the model')
        # parser.add_argument('--class_weights', type=str, default="[0.15, 0.85]", help='Class weights as JSON string')
        parser.add_argument('--use_sampler', type=lambda x: x.lower() == 'true', default=True, help='Use weighted sampler')
        parser.add_argument('--threshold', type=float, default=0.85, help='Confidence threshold for predictions')
        
        # Data configuration
        parser.add_argument('--val_split', type=float, default=0.2, help='Validation set size ratio')
        parser.add_argument('--test_size', type=float, default=0.15, help='Test set size ratio')
        parser.add_argument('--augmentation_intensity', type=float, default=0.8, 
                          help='Intensity of data augmentation (0.0 to 1.0)')
        
        # Early stopping
        parser.add_argument('--early_stopping', type=lambda x: x.lower() == 'true', default=True,
                          help='Enable early stopping')
        parser.add_argument('--patience', type=int, default=10, 
                          help='Number of epochs to wait before early stopping')
        parser.add_argument('--min_delta', type=float, default=0.001, 
                          help='Minimum change in monitored metric to qualify as improvement')
        parser.add_argument('--monitor_metric', type=str, default='f1_anormal',
                          help='Metric to monitor for early stopping')
        
        # Paths
        parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'),
                          help='Path to training data directory')
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './models'),
                          help='Directory to save trained models')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed for reproducibility')
        
        return parser.parse_args()