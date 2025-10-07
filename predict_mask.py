import argparse
from TFNet import predict_binary_mask


def main():
    parser = argparse.ArgumentParser(description='TF-Net Binary Mask Prediction')

    parser.add_argument('--model_path', type=str, help='Path to the trained model file')
    parser.add_argument('--image_dir', type=str, help='Directory containing input images')
    parser.add_argument('--output_path', type=str, help='Directory to save output masks')

    args_ = parser.parse_args()

    return args_


if __name__ == '__main__':
    args = main()
    
    if not any(vars(args).values()):
        predict_binary_mask(
            model_path='path/to/model',  # e.g. ./models/TFNet_2_1.0_False_512-mobileone-mini.pt
            image_dir='path/to/TF-crop-test',
            output_path='path/to/output_path'
        )
    else:
        predict_binary_mask(
            model_path=args.model_path,
            image_dir=args.image_dir,
            output_path=args.output_path
        )
