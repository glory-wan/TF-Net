import argparse
from TFNet import train_model


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Image Segmentation Training')

    # Add arguments corresponding to train_model parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory (required)')
    parser.add_argument('--base_model_name', type=str, default='TFNet',
                        help='Name of the base model architecture')

    # model arguments
    parser.add_argument('--nc', '--num_class', type=int, default=2,
                        help='Number of input channels for the model')
    parser.add_argument('--wid', type=float, default=1.0,
                        help='Width multiplier for model scaling')
    parser.add_argument('--imgz', type=int, default=512,
                        help='Input image size (height and width)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Number of samples per batch')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of segmentation classes')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='Momentum factor for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 penalty) for optimizer')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Factor by which learning rate is reduced')
    parser.add_argument('--patience', type=int, default=3,
                        help='Epochs to wait before reducing learning rate')

    args = parser.parse_args()

    # Call train_model with parsed arguments
    train_model(
        base_model_name=args.base_model_name,
        nc=args.nc,
        wid=args.wid,
        imgz=args.imgz,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        lr=args.lr,
        epochs=args.epochs,
        workers=args.workers,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        factor=args.factor,
        patience=args.patience
    )


if __name__ == '__main__':
    main()
