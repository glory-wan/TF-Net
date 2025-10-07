import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import numpy as np
import json
import gc
import os

from TFNet.utils import calculate_iou, calculate_confusion_matrix, calculate_class_weights, set_seed
from TFNet.TFMDataset import load_seg_datasets
from TFNet import create_tfnet_variants


class Trainer:
    """Training class for TF-Net segmentation model"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, width,
                 num_epochs, device, num_classes, backbone='mobileone-mini', model_name='TF-Net', save_dir='./models'):
        """
        Initialize the trainer

        Args:
            model: The neural network model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimization algorithm
            scheduler: Learning rate scheduler
            num_epochs: Number of training epochs
            device: Training device (cuda/cpu)
            num_classes: Number of segmentation classes
            backbone: Backbone network name
            model_name: Model identifier name
            save_dir: Directory to save models and results
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.num_classes = num_classes
        self.backbone = backbone
        self.model_name = model_name
        self.save_dir = save_dir
        self.width = width

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'train_json'), exist_ok=True)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []

        # Best metrics tracking
        self.best_val_loss = float('inf')
        self.best_miou = 0.0
        self.best_iou_per_class = [0.0] * num_classes
        self.best_epoch = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Train]')

        for images, masks in train_pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).squeeze(1)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return running_loss / len(self.train_loader)

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        val_pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs} [Val]')

        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).squeeze(1)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()

                # Get predictions
                preds = torch.argmax(outputs, dim=1)

                # Collect predictions and targets for mIoU calculation
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate mIoU
        miou, iou_per_class = self.calculate_metrics(all_preds, all_targets)

        return val_loss / len(self.val_loader), miou, iou_per_class

    def calculate_metrics(self, all_preds, all_targets):
        """Calculate mIoU and per-class IoU"""
        if not all_preds or not all_targets:
            return 0.0, [0.0] * self.num_classes

        all_preds_np = np.concatenate(all_preds)
        all_targets_np = np.concatenate(all_targets)

        # Manually calculate confusion matrix
        conf_matrix = calculate_confusion_matrix(all_preds_np, all_targets_np, self.num_classes)
        miou, iou_per_class = calculate_iou(conf_matrix)

        return miou, iou_per_class

    def save_checkpoint(self, epoch, miou, iou_per_class, val_loss):
        """Save model checkpoint if it's the best so far"""
        if miou > self.best_miou:
            self.best_miou = miou
            self.best_iou_per_class = iou_per_class
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.best_val_loss,
                'miou': self.best_miou,
                'nc': self.num_classes,
                'wid': self.width,
            }

            model_path = os.path.join(self.save_dir, f'{self.model_name}-{self.backbone}.pt')
            torch.save(checkpoint, model_path)
            print(f"Best model saved with mIoU: {miou:.4f}")

    def save_results(self):
        """Save training results to JSON file"""
        results_dict = {
            'model_name': self.model_name,
            'backbone': self.backbone,
            'best_epoch': self.best_epoch,
            'best_miou': self.best_miou,
            'best_iou_per_class': self.best_iou_per_class.tolist() if hasattr(self.best_iou_per_class, 'tolist')
            else list(self.best_iou_per_class),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_mious': self.val_mious,
        }

        json_path = os.path.join(self.save_dir, 'train_json', f'{self.model_name}_{self.backbone}.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)

        print('#' * 10, f'Results saved to {json_path}', '#' * 10)

    def print_epoch_summary(self, epoch, train_loss, val_loss, miou, iou_per_class):
        """Print summary for current epoch"""
        print(f'Epoch {epoch + 1}/{self.num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'mIoU: {miou:.4f}')

        for i, iou in enumerate(iou_per_class):
            print(f'  Class {i} IoU: {iou:.4f}')

    def print_final_summary(self):
        """Print final training summary"""
        print(f'\nModel: {self.model_name}, Backbone: {self.backbone}')
        print(f'Best model in epoch {self.best_epoch} with '
              f'mIoU: {self.best_miou:.3f}, '
              f'Val Loss: {self.best_val_loss:.4f}')

        for i, iou in enumerate(self.best_iou_per_class):
            print(f'  Class {i} IoU: {iou:.3f}')

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.model_name} with backbone {self.backbone}")
        print(f"Training for {self.num_epochs} epochs on {self.device}")

        for epoch in range(self.num_epochs):
            # Training phase
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss, miou, iou_per_class = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            self.val_mious.append(miou)

            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Print epoch summary
            self.print_epoch_summary(epoch, train_loss, val_loss, miou, iou_per_class)

            # Save checkpoint if best model
            self.save_checkpoint(epoch, miou, iou_per_class, val_loss)

        # Save final results
        self.save_results()
        self.print_final_summary()

        return self.train_losses, self.val_losses, self.val_mious


def train_model(
        base_model_name='TFNet',
        nc=2,
        wid=1.0,
        se=False,
        imgz=512,
        data_dir=None,
        batch_size=2,
        num_classes=2,
        lr=0.001,
        epochs=50,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        workers=8,
        momentum=0.937,
        weight_decay=5 * 1e-4,
        factor=0.5,
        patience=3,
):
    """
    Train a segmentation model with customizable parameters.

    Args:
        base_model_name (str): Name of the base model architecture. Default: 'TFNet'
        nc (int): Number of input channels for the model. Default: 2
        wid (float): Width multiplier for model scaling. Default: 1.0
        se (bool): Whether to use Squeeze-and-Excitation blocks. Default: True
        imgz (int): Input image size (height and width). Default: 512
        data_dir (str): Path to the dataset directory. Must be provided.
        batch_size (int): Number of samples per batch. Default: 2
        num_classes (int): Number of segmentation classes. Default: 2
        lr (float): Initial learning rate for optimizer. Default: 0.001
        epochs (int): Number of training epochs. Default: 50
        device (torch.device): Device for training (CPU/GPU). Default: CUDA if available, else CPU
        workers (int): Number of data loading workers. Default: 8
        momentum (float): Momentum factor for SGD optimizer. Default: 0.937
        weight_decay (float): Weight decay (L2 penalty) for optimizer. Default: 5e-4
        factor (float): Factor by which learning rate is reduced. Default: 0.5
        patience (int): Number of epochs with no improvement after which learning rate will be reduced. Default: 3

    Returns:
        dict: Dictionary containing training results with keys:
            - train_losses (list): Training losses for each epoch
            - val_losses (list): Validation losses for each epoch
            - val_mious (list): Validation mIoU scores for each epoch
            - model_name (str): Generated name of the trained model
            - best_miou (float): Best validation mIoU achieved during training

    Raises:
        ValueError: If data_dir is not provided

    Example:
        >>> results = train_model(
        ...     data_dir='/path/to/dataset',
        ...     base_model_name='TFNet',
        ...     num_classes=2,
        ...     epochs=100
        ... )
        >>> print(f"Best mIoU: {results['best_miou']:.4f}")
    """
    if data_dir is None:
        raise ValueError("data_dir must be provided")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed()

    # Create model instance
    models_ = create_tfnet_variants(
        num_classes=nc,
        width=wid,
        inference_mode=False
    )
    model = models_['full']
    model.to(device)

    # Generate model identifier
    model_name = f'{base_model_name}_{nc}_{wid}_{se}_{imgz}'
    print('*' * 20, f'{model_name} -- {data_dir}', '*' * 20)

    # Load and prepare datasets
    TFMSegLoaders = load_seg_datasets(data_dir, batch_size, workers=workers, imgz=imgz)
    class_counts, class_weights = calculate_class_weights(
        TFMSegLoaders['train'], num_classes, device
    )

    # Setup loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=TFMSegLoaders['train'],
        val_loader=TFMSegLoaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=epochs,
        device=device,
        num_classes=num_classes,
        backbone='mobileone-mini',
        model_name=model_name,
        width=wid,
    )

    # Execute training process
    train_losses, val_losses, val_mious = trainer.train()

    # Print final results
    print(f'Final Validation mIoU: {val_mious[-1]:.4f}')
    print(f'Best Validation mIoU: {max(val_mious):.4f} \n')

    # Cleanup resources
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_mious': val_mious,
        'model_name': model_name,
        'best_miou': max(val_mious)
    }
