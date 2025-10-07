from TFNet import simple_evaluate_folders
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate segmentation performance by computing metrics '
                    'between ground truth and predicted masks. '
                    'Metrics include: IoU, DSC (Dice), Recall, FPR, HD95, ASSD '
                    'for both background and foreground classes, plus their mean values.')

    parser.add_argument('--gt_folder', type=str, help='Directory containing ground truth mask images')
    parser.add_argument('--pred_folder', type=str, help='Directory containing predicted mask images')
    parser.add_argument('--save_folder', type=str, help='Output directory for evaluation results CSV')

    args_ = parser.parse_args()

    return args_


if __name__ == '__main__':
    args = main()

    if not any(vars(args).values()):
        simple_evaluate_folders(
            gt_folder='path/to/gt',
            pred_folder='/path/to/predict_mask',
            save_folder='path/to/save/metrics',
        )
    else:
        simple_evaluate_folders(
            gt_folder=args.gt_folder,
            pred_folder=args.pred_folder,
            save_folder=args.save_folder,
        )
