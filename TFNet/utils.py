import torch
import numpy as np
import random
import torch.nn as nn
from tqdm import tqdm
import os


def set_seed(seed=42):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def calculate_class_weights(loader, num_classes, device):
    """Calculate class weights based on pixel frequency in dataset

    Args:
        loader: DataLoader for the dataset
        num_classes: Number of classes in segmentation
        device: Target device for the weights tensor

    Returns:
        class_counts: Array of pixel counts per class
        weights: Normalized class weights tensor
    """
    class_counts = np.zeros(num_classes)
    for _, masks in tqdm(loader, desc="Calculating class weights", unit="batch"):
        masks_np = masks.numpy()
        for i in range(num_classes):
            class_counts[i] += np.sum(masks_np == i)

    total_pixels = np.sum(class_counts)
    print("Class distribution:")
    for i in range(num_classes):
        percentage = (class_counts[i] / total_pixels) * 100
        print(f"Class {i}: pixels percentage {percentage:.2f}%")

    # Calculate weights (inverse of class frequency)
    weights = np.sum(class_counts) / (num_classes * class_counts)
    weights = weights / np.sum(weights)  # Normalize
    return class_counts, torch.tensor(weights, dtype=torch.float32).to(device)


def calculate_confusion_matrix(preds, targets, num_classes):
    """
    Manually calculate confusion matrix for segmentation tasks

    Args:
        preds: [N, H, W] predicted labels
        targets: [N, H, W] ground truth labels
        num_classes: Number of classes

    Returns:
        conf_matrix: [num_classes, num_classes] confusion matrix
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(num_classes):
        for j in range(num_classes):
            # Count pixels where ground truth is i and prediction is j
            mask = (targets == i) & (preds == j)
            conf_matrix[i, j] = np.sum(mask)

    return conf_matrix


def calculate_iou(conf_matrix):
    """
    Calculate IoU for each class and mean IoU

    Args:
        conf_matrix: [num_classes, num_classes] confusion matrix

    Returns:
        miou: Mean Intersection over Union
        iou_per_class: IoU values for each class
    """
    iou_per_class = np.zeros(conf_matrix.shape[0])
    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]  # True positives
        fp = np.sum(conf_matrix[:, i]) - tp  # False positives
        fn = np.sum(conf_matrix[i, :]) - tp  # False negatives

        if (tp + fp + fn) > 0:
            iou_per_class[i] = tp / (tp + fp + fn)
        else:
            iou_per_class[i] = 0.0

    miou = np.mean(iou_per_class)

    return miou, iou_per_class
