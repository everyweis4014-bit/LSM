import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class F1ScoreCalculator:
    """Class to calculate F1 scores for binary and multi-class classification"""
    
    @staticmethod
    def calculate_f1_score(y_true, y_pred, average='weighted'):
        """
        Calculate F1 score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: 'weighted', 'macro', 'micro', or None
            
        Returns:
            F1 score value
        """
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    
    @staticmethod
    def calculate_precision(y_true, y_pred, average='weighted'):
        """
        Calculate precision score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: 'weighted', 'macro', 'micro', or None
            
        Returns:
            Precision score value
        """
        return precision_score(y_true, y_pred, average=average, zero_division=0)
    
    @staticmethod
    def calculate_recall(y_true, y_pred, average='weighted'):
        """
        Calculate recall score
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: 'weighted', 'macro', 'micro', or None
            
        Returns:
            Recall score value
        """
        return recall_score(y_true, y_pred, average=average, zero_division=0)
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, average='weighted'):
        """
        Calculate F1, Precision, and Recall scores
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: 'weighted', 'macro', 'micro', or None
            
        Returns:
            Dictionary containing F1, Precision, and Recall scores
        """
        return {
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0)
        }


class TrainingHistory:
    """Class to track and record training metrics including F1 scores"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    
    def update(self, **kwargs):
        """
        Update history with new metrics
        
        Args:
            **kwargs: Key-value pairs of metrics to update
        """
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_history(self):
        """Return the complete history dictionary"""
        return self.history
    
    def log_epoch(self, epoch, metrics):
        """
        Log epoch metrics to logger
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics to log
        """
        log_msg = f"Epoch {epoch} - "
        log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(log_msg)


class CNNTrainer:
    """Main trainer class for CNN model"""
    
    def __init__(self, model, device, criterion=None, optimizer=None):
        """
        Initialize the trainer
        
        Args:
            model: CNN model to train
            device: Device to train on (cpu or cuda)
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
        """
        self.model = model
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.history = TrainingHistory()
        self.f1_calculator = F1ScoreCalculator()
        logger.info(f"Trainer initialized on device: {device}")
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary containing loss and F1 metrics
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions and targets for F1 calculation
            preds = output.argmax(dim=1).cpu().numpy()
            targets = target.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
        
        avg_loss = total_loss / len(train_loader)
        
        # Calculate F1 and related metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = self.f1_calculator.calculate_all_metrics(all_targets, all_preds)
        accuracy = np.mean(all_preds == all_targets)
        
        metrics['loss'] = avg_loss
        metrics['accuracy'] = accuracy
        
        logger.debug(f"Training metrics - Loss: {avg_loss:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics, avg_loss, all_preds, all_targets
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary containing loss and F1 metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                # Collect predictions and targets for F1 calculation
                preds = output.argmax(dim=1).cpu().numpy()
                targets = target.cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets)
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate F1 and related metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        metrics = self.f1_calculator.calculate_all_metrics(all_targets, all_preds)
        accuracy = np.mean(all_preds == all_targets)
        
        metrics['loss'] = avg_loss
        metrics['accuracy'] = accuracy
        
        logger.debug(f"Validation metrics - Loss: {avg_loss:.4f}, F1: {metrics['f1']:.4f}")
        
        return metrics, avg_loss, all_preds, all_targets
    
    def fit(self, train_loader, val_loader, epochs, scheduler=None):
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Train epoch
            train_metrics, train_loss, train_preds, train_targets = self.train_epoch(train_loader)
            
            # Validate
            val_metrics, val_loss, val_preds, val_targets = self.validate(val_loader)
            
            # Update history
            self.history.update(
                train_loss=train_loss,
                val_loss=val_loss,
                train_f1=train_metrics['f1'],
                val_f1=val_metrics['f1'],
                train_precision=train_metrics['precision'],
                val_precision=val_metrics['precision'],
                train_recall=train_metrics['recall'],
                val_recall=val_metrics['recall'],
                train_accuracy=train_metrics['accuracy'],
                val_accuracy=val_metrics['accuracy']
            )
            
            # Log epoch metrics
            epoch_log_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_f1': train_metrics['f1'],
                'val_f1': val_metrics['f1'],
                'train_acc': train_metrics['accuracy'],
                'val_acc': val_metrics['accuracy'],
                'train_precision': train_metrics['precision'],
                'val_precision': val_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'val_recall': val_metrics['recall']
            }
            self.history.log_epoch(epoch, epoch_log_metrics)
            
            # Step scheduler if provided
            if scheduler:
                scheduler.step()
            
            # Print epoch summary
            print(f"Epoch {epoch}: Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}, "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        logger.info("Training completed")
        return self.history.get_history()
    
    def get_history(self):
        """Return training history"""
        return self.history.get_history()


def save_training_history(history, filename='training_history.txt'):
    """
    Save training history to a file
    
    Args:
        history: Training history dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write(f"Training History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for epoch in range(len(history['train_loss'])):
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Train Loss: {history['train_loss'][epoch]:.4f}\n")
            f.write(f"  Val Loss: {history['val_loss'][epoch]:.4f}\n")
            f.write(f"  Train F1: {history['train_f1'][epoch]:.4f}\n")
            f.write(f"  Val F1: {history['val_f1'][epoch]:.4f}\n")
            f.write(f"  Train Accuracy: {history['train_accuracy'][epoch]:.4f}\n")
            f.write(f"  Val Accuracy: {history['val_accuracy'][epoch]:.4f}\n")
            f.write(f"  Train Precision: {history['train_precision'][epoch]:.4f}\n")
            f.write(f"  Val Precision: {history['val_precision'][epoch]:.4f}\n")
            f.write(f"  Train Recall: {history['train_recall'][epoch]:.4f}\n")
            f.write(f"  Val Recall: {history['val_recall'][epoch]:.4f}\n")
            f.write("-" * 40 + "\n")
    
    logger.info(f"Training history saved to {filename}")


# Example usage
if __name__ == "__main__":
    logger.info("CNN Training Script Initialized")
    # Example configuration can be added here
