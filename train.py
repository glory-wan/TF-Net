from TFNet import train_model


def main():
    """Main training script - direct execution version"""
    # Set your parameters directly here
    data_dir = 'path/to/TF-crop dataset'  # Change this to your dataset path

    # Call train_model with direct parameters
    train_model(
        base_model_name='TFNet',
        nc=2,
        wid=0.25,
        imgz=512,
        data_dir=data_dir,
        batch_size=2,
        num_classes=2,
        lr=0.001,
        epochs=50,
        workers=8,
        momentum=0.937,
        weight_decay=5e-4,
        factor=0.5,
        patience=3
    )


if __name__ == '__main__':
    main()
