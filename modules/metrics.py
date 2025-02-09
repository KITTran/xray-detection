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
    assert isinstance(predictions, torch.Tensor), "Prediction must be a PyTorch tensor."

    predictions[predictions < threshold] = 0
    predictions[predictions > threshold] = 1

    intersection = torch.sum(predictions * targets).float()
    total_positive = torch.sum(targets).float()

    return (intersection / (total_positive + 1e-7)).item()


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

class UnetMetrics:
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

        dice = dice_coefficient(predictions.squeeze(), targets.squeeze())
        sens = sensitivity(predictions.squeeze(), targets.squeeze(), self.threshold)

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

        # Specificity
        specificity = tn / (tn + fp + 1e-7)

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)

        # Jaccard Index
        jaccard = tp / (tp + fp + fn + 1e-7)

        # AUC Score
        try:
            auc = roc_auc_score(targets, predictions)
        except ValueError:
            auc = 0.0  # Handle cases where AUC cannot be computed

        # Precision-Recall Curve
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        metrics = {
            'accuracy': accuracy,
            'sensitivity': sens,
            'specificity': specificity,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'dice': dice,
            'jaccard': jaccard
        }
        return metrics

class ResMetrics:
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

        dice = dice_coefficient(predictions.squeeze(), targets.squeeze(), threshold=self.threshold)
        sens = sensitivity(predictions.squeeze(), targets.squeeze(), self.threshold)

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

        # Jaccard Index
        jaccard = tp / (tp + fp + fn + 1e-7)

        # recall
        recall = tp / (tp + fn + 1e-7)

        # precision
        precision = tp / (tp + fp + 1e-7)


        metrics = {
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'jaccard': jaccard,
            'sensitivity': sens
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
    input_tensor = torch.randn(100, 1, 128, 128)
    target_tensor = torch.randint(0, 2, (100, 1, 128, 128))

    dataloader = torch.utils.data.DataLoader(list(zip(input_tensor, target_tensor)), batch_size=10)

    metrics = ResMetrics(threshold=0.7)

    torch.autograd.set_detect_anomaly(True)
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        train_metrics = {}

        for inputs, targets in dataloader:
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets.float())

            # calculate metrics
            metrics = metrics.compute_metrics(outputs, targets)
            try:
                for key in train_metrics.keys():
                    train_metrics[key] += metrics[key]
            except:
                train_metrics = metrics

            loss.backward()
            optimizer.step()

        # Average metrics
        for key in train_metrics.keys():
            train_metrics[key] /= len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        print(f"Dice Coefficient: {train_metrics['dice']}")
        print(f"Precision-Recall Curve: {train_metrics['precision']}, {train_metrics['recall']}")
        print(f"Sensitivity: {train_metrics['sensitivity']}")
        # print(f"Specificity: {train_metrics['specificity']}")
        # print(f"Accuracy: {train_metrics['accuracy']}")
        # print(f"AUC Score: {train_metrics['auc']}")
        print('Jaccard Index: ', train_metrics['jaccard'])
        print('-' * 50)

        with torch.no_grad():
            # Get predictions
            predictions = model(input_tensor)

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
