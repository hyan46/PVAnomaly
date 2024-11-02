import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

class CNN(nn.Module):
    """
    Convolutional Neural Network for fault detection and prognostics
    Handles both current state detection and future state prediction
    """
    def __init__(self, task='detection'):
        """
        Initialize the CNN model
        Args:
            task (str): Either 'detection' or 'prognostics'
        """
        super(CNN, self).__init__()
        self.task = task
        
        # First convolutional block: 1->64 channels, 5x2 kernel
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(5,2),
                stride=2,
                padding=1,
            ),
            nn.MaxPool2d(kernel_size=2),
        )
        
        # Second convolutional block: 64->32 channels, 5x2 kernel
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, (5,2), 1, 1),
            nn.MaxPool2d(2),
        )
        
        # Output layers for different tasks
        self.detection_out = nn.Linear(352, 4)  # 4 fault types
        if task == 'prognostics':
            self.prognostic_out = nn.Linear(352, 4)  # Future fault prediction

    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, height, width]
        Returns:
            torch.Tensor or tuple: Output predictions (detection) or (current, future) predictions (prognostics)
        """
        # Pass through convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        
        if self.task == 'detection':
            return self.detection_out(x)
        else:
            current = self.detection_out(x)
            future = self.prognostic_out(x)
            return current, future

class ModelTrainer:
    """
    Trainer class for handling model training, testing, and evaluation
    """
    def __init__(self, model, device='cuda:3', task='detection'):
        """
        Initialize the trainer
        Args:
            model: CNN model instance
            device: Computing device (CPU/GPU)
            task: 'detection' or 'prognostics'
        """
        self.model = model.to(device)
        self.device = device
        self.task = task
        # Binary cross entropy loss with positive weight for imbalanced classes
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5]).to(device))
        
    def train(self, train_loader, epochs=150):
        """
        Train the model
        Args:
            train_loader: DataLoader for training data
            epochs: Number of training epochs
        """
        for epoch in range(epochs):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05/(epoch+1))
            train_loss = []
            
            for step, data in enumerate(train_loader):
                # Handle different data formats for detection/prognostics
                if self.task == 'detection':
                    b_x, b_y = data
                    output = self.model(b_x.unsqueeze(1))
                    loss = self.loss_func(output, b_y)
                else:
                    b_x, b_y, b_future = data
                    current, future = self.model(b_x.unsqueeze(1))
                    loss = self.loss_func(current, b_y) + self.loss_func(future, b_future)
                
                # Backpropagation
                train_loss.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            avg_loss = sum(train_loss)/len(train_loader)
            print(f'Epoch: {epoch} loss: {avg_loss}')
    
    def test(self, test_loader, batch_size=10000):
        """
        Test the model
        Args:
            test_loader: DataLoader for test data
            batch_size: Batch size for testing
        Returns:
            tuple: Predictions and targets (format depends on task)
        """
        test_loss = 0.0
        all_current_predictions = []
        all_current_targets = []
        all_future_predictions = []
        all_future_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                # Handle different data formats for detection/prognostics
                if self.task == 'detection':
                    x, target = data
                    output = self.model(x.unsqueeze(1))
                    loss = self.loss_func(output, target)
                    
                    probs = torch.sigmoid(output)
                    predictions = (probs > 0.5).float()
                    all_current_predictions.append(predictions)
                    all_current_targets.append(target)
                else:
                    x, current_target, future_target = data
                    current, future = self.model(x.unsqueeze(1))
                    loss = self.loss_func(current, current_target) + self.loss_func(future, future_target)
                    
                    # Convert logits to probabilities and then to binary predictions
                    current_probs = torch.sigmoid(current)
                    future_probs = torch.sigmoid(future)
                    current_predictions = (current_probs > 0.5).float()
                    future_predictions = (future_probs > 0.5).float()
                    
                    all_current_predictions.append(current_predictions)
                    all_current_targets.append(current_target)
                    all_future_predictions.append(future_predictions)
                    all_future_targets.append(future_target)
                
                test_loss += loss.item() * x.size(0)
        
        test_loss_avg = test_loss/len(test_loader.dataset)
        print(f'Test loss: {test_loss_avg}')
        
        # Return appropriate format based on task
        if self.task == 'detection':
            return (torch.cat(all_current_predictions), 
                    torch.cat(all_current_targets))
        else:
            return (torch.cat(all_current_predictions), 
                    torch.cat(all_current_targets),
                    torch.cat(all_future_predictions), 
                    torch.cat(all_future_targets))
    
    def evaluate_metrics(self, predictions, targets, save_path=None, prefix=''):
        """
        Calculate and save evaluation metrics
        Args:
            predictions: Model predictions
            targets: True labels
            save_path: Path to save visualization results
            prefix: Prefix for saved files
        Returns:
            tuple: F1 scores, confusion matrices, overall F1, overall confusion matrix
        """
        f1_scores = []
        confusion_matrices = []
        
        # Calculate metrics for each fault type
        for i in range(4):
            f1 = f1_score(targets[:,i].cpu(), predictions[:,i].cpu())
            cf = confusion_matrix(targets[:,i].cpu(), predictions[:,i].cpu())
            f1_scores.append(f1)
            confusion_matrices.append(cf)
            print(f'Fault {i} F1 score: {f1}')
            print(f'Confusion Matrix:\n{cf}')
            
            # Save confusion matrix plots
            if save_path:
                plt.figure(figsize=(4, 3), dpi=180)
                plt.tight_layout()
                sns.heatmap(cf, annot=True, fmt=".0f")
                plt.savefig(Path(save_path, f'{prefix}fault_{i}.png'))
                plt.close()
        
        # Calculate overall metrics
        binary_true = (targets.mean(dim=1) > 0).float()
        binary_pred = (predictions.mean(dim=1) > 0).float()
        overall_f1 = f1_score(binary_true.cpu(), binary_pred.cpu())
        overall_cf = confusion_matrix(binary_true.cpu(), binary_pred.cpu())
        
        if save_path:
            plt.figure(figsize=(4, 3), dpi=220)
            sns.heatmap(overall_cf, annot=True, fmt=".0f")
            plt.savefig(Path(save_path, f'{prefix}overall.png'))
            plt.close()
        
        return f1_scores, confusion_matrices, overall_f1, overall_cf
    
    def save_model(self, path):
        """Save model state dict to file"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model state dict from file"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

# Main execution block
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Test both detection and prognostics tasks
    for task in ['detection', 'prognostics']:
        print(f"\nRunning {task} task...")
        
        # Data preparation
        data_loader = DataLoader(task=task)
        train_loader, test_loader, le = data_loader.prepare_data()
        
        # Model initialization and training
        model = CNN(task=task)
        trainer = ModelTrainer(model, task=task)
        trainer.train(train_loader)
        
        # Testing and evaluation
        if task == 'detection':
            predictions, targets = trainer.test(test_loader)
            save_path = f'../Multilabel_classification/result_img/report/{task}'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            f1_scores, cms, overall_f1, overall_cm = trainer.evaluate_metrics(
                predictions, targets, save_path=save_path
            )
        else:
            # Handle both current and future predictions for prognostics
            current_pred, current_targets, future_pred, future_targets = trainer.test(test_loader)
            save_path = f'../Multilabel_classification/result_img/report/{task}'
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            print("\nCurrent State Metrics:")
            f1_scores, cms, overall_f1, overall_cm = trainer.evaluate_metrics(
                current_pred, current_targets, save_path=save_path, prefix='current_'
            )
            
            print("\nFuture State Metrics:")
            f1_scores, cms, overall_f1, overall_cm = trainer.evaluate_metrics(
                future_pred, future_targets, save_path=save_path, prefix='future_'
            )
        
        # Save trained model
        trainer.save_model(f'saved_models/{task}_model.pth')