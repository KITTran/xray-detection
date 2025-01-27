import torch
from sklearn.metrics import roc_auc_score

def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

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

if __name__ == "__main__":
    # Example Usage
    predictions = torch.tensor([0.1, 0.4, 0.35, 0.8], requires_grad=False)
    targets = torch.tensor([0, 0, 1, 1])

    # Compute metrics
    sens = sensitivity(predictions, targets)
    spec = specificity(predictions, targets)
    auc = auc_score(predictions, targets)
    dice  = dice_coefficient(predictions, targets)

    print(f"Sensitivity: {sens}")
    print(f"Specificity: {spec}")
    print(f"AUC: {auc}")
    print(f"Dice Coefficient: {dice}")
