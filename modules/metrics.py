import torch
import torchmetrics
from sklearn.metrics import roc_auc_score, precision_recall_curve

import torch.nn as nn
import torch.optim as optim

def dice_coefficient(prediction, target, epsilon=1e-07, threshold=0.7):
    # Convert from numpy to tensor
    assert isinstance(prediction, torch.Tensor), "Prediction must be a PyTorch tensor."

    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < threshold] = 0
    prediction_copy[prediction_copy > threshold] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice

def sensitivity(predictions, targets, threshold=0.5):
    """
    Compute Sensitivity (Recall or True Positive Rate).

    Parameters:
    - predictions: torch.Tensor, predicted probabilities.
    - targets: torch.Tensor, ground truth labels.
    - threshold: float, decision threshold for classification.

    Returns:
    - Sensitivity: float.
    """
    # Binarize predictions based on threshold
    preds = (predictions >= threshold).int()
    true_positive = torch.sum((preds == 1) & (targets == 1)).float()
    false_negative = torch.sum((preds == 0) & (targets == 1)).float()
    return (true_positive / (true_positive + false_negative)).item()

def specificity(predictions, targets, threshold=0.5):
    """
    Compute Specificity (True Negative Rate).

    Parameters:
    - predictions: torch.Tensor, predicted probabilities.
    - targets: torch.Tensor, ground truth labels.
    - threshold: float, decision threshold for classification.

    Returns:
    - Specificity: float.
    """
    # Binarize predictions based on threshold
    preds = (predictions >= threshold).int()
    true_negative = torch.sum((preds == 0) & (targets == 0)).float()
    false_positive = torch.sum((preds == 1) & (targets == 0)).float()
    return (true_negative / (true_negative + false_positive)).item()

def auc_score(predictions, targets):
    """
    Compute Area Under the Curve (AUC).

    Parameters:
    - predictions: torch.Tensor, predicted probabilities.
    - targets: torch.Tensor, ground truth labels.

    Returns:
    - AUC: float.
    """
    # Convert to NumPy arrays for compatibility with sklearn
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    return roc_auc_score(targets_np, predictions_np)

class Metrics:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def compute_metrics(self, predictions, targets):
        """
        Compute binary classification metrics for semantic segmentation.

        Args:
            predictions: Predicted logits or probabilities (B, H, W).
            targets: Ground truth binary masks (B, H, W).

        Returns:
            Dictionary of metrics (accuracy, sensitivity, specificity, auc, dice).
        """
        # # Apply sigmoid
        # predictions = torch.sigmoid(predictions)

        # Flatten tensors
        predictions = predictions.view(-1).detach().cpu().numpy()
        targets = targets.view(-1).detach().cpu().numpy()

        # Threshold to binary predictions
        pred_labels = (predictions > self.threshold).astype(int)

        # True positives, false positives, true negatives, false negatives
        tp = (pred_labels * targets).sum()
        fp = (pred_labels * (1 - targets)).sum()
        tn = ((1 - pred_labels) * (1 - targets)).sum()
        fn = ((1 - pred_labels) * targets).sum()

        # Sensitivity (Recall)
        sensitivity = tp / (tp + fn + 1e-7)

        # Specificity
        specificity = tn / (tn + fp + 1e-7)

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)

        # AUC Score
        try:
            auc = roc_auc_score(targets, predictions)
        except ValueError:
            auc = 0.0  # Handle cases where AUC cannot be computed

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(targets, predictions, pos_label=1)

        dice = dice_coefficient(predictions, targets)

        metrics = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'dice': dice
        }
        return metrics

if __name__ == "__main__":
    # Define a simple binary semantic segmentation model
    class SimpleSegmentationModel(nn.Module):
        def __init__(self):
            super(SimpleSegmentationModel, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    # Create model, define loss function and optimizer
    model = SimpleSegmentationModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate dummy data
    input_tensor = torch.randn(8, 1, 128, 128)
    target_tensor = torch.randint(0, 2, (8, 1, 128, 128))

    metrics_calculator = Metrics(threshold=0.7)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        predictions = model(input_tensor)
        loss = criterion(predictions, target_tensor.float())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate metrics
        # prc = prc(predictions, target_tensor)
        # sen = sen(predictions, target_tensor)
        # spe = spe(predictions, target_tensor)
        # acc = acc(predictions, target_tensor)
        # auc = auc(predictions, target_tensor)

        dice = dice_coefficient(predictions, target_tensor)
        metrics = metrics_calculator.compute_metrics(predictions, target_tensor)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        print(f"Dice Coefficient: {dice}")
        print(f"Precision-Recall Curve: {metrics['precision']}, {metrics['recall']}")
        print(f"Sensitivity: {metrics['sensitivity']}")
        print(f"Specificity: {metrics['specificity']}")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"AUC Score: {metrics['auc']}")
        print('-' * 50)

    # Visualize prediction and target
    import matplotlib.pyplot as plt
    plt.imshow(predictions[0].squeeze().detach().cpu().numpy(), cmap='gray')
    plt.title('Prediction')
    plt.show()

    plt.imshow(target_tensor[0].squeeze().detach().cpu().numpy(), cmap='gray')
    plt.title('Target')
    plt.show()

    # Viualize precision-recall curve
    plt.plot(metrics['recall'], metrics['precision'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
