import os
import numpy as np
import cv2
from medpy.metric.binary import dc, jc, recall, hd95, assd
import pandas as pd
from tqdm import tqdm


def calculate_metrics_multiclass(gt, pred, num_classes=2):
    """Calculate all metrics for multiple classes"""
    metrics = {}

    for class_id in range(num_classes):
        gt_class = (gt == class_id)
        pred_class = (pred == class_id)

        gt_bool = gt_class.astype(bool)
        pred_bool = pred_class.astype(bool)

        intersection = np.logical_and(gt_class, pred_class).sum()
        union = np.logical_or(gt_class, pred_class).sum()
        iou = intersection / union if union != 0 else 0.0

        dice = (2.0 * intersection) / (gt_class.sum() + pred_class.sum()) if (
                                                                                         gt_class.sum() + pred_class.sum()) != 0 else 0.0

        tp = intersection
        fn = np.logical_and(gt_class, np.logical_not(pred_class)).sum()
        recall_val = tp / (tp + fn) if (tp + fn) != 0 else 0.0

        fp = np.logical_and(np.logical_not(gt_class), pred_class).sum()
        tn = np.logical_and(np.logical_not(gt_class), np.logical_not(pred_class)).sum()
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

        if class_id == 0:
            prefix = 'Background'
        else:
            prefix = 'Class1'

        metrics[f'{prefix}_IoU'] = iou
        metrics[f'{prefix}_DSC'] = dice
        metrics[f'{prefix}_Recall'] = recall_val
        metrics[f'{prefix}_FPR'] = fpr

    background_iou = metrics['Background_IoU']
    class1_iou = metrics['Class1_IoU']
    metrics['mIoU'] = (background_iou + class1_iou) / 2
    metrics['mDSC'] = (metrics['Background_DSC'] + metrics['Class1_DSC']) / 2
    metrics['mRecall'] = (metrics['Background_Recall'] + metrics['Class1_Recall']) / 2
    metrics['mFPR'] = (metrics['Background_FPR'] + metrics['Class1_FPR']) / 2

    return metrics


def simple_evaluate_folders(gt_folder, pred_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    results = []
    gt_files = [f for f in os.listdir(gt_folder) if f.lower().endswith('.png')]

    for gt_file in tqdm(gt_files, desc=f'processing {os.path.basename(pred_folder)}'):
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, gt_file)

        if not os.path.exists(pred_path):
            print(f"Skipping {pred_path}: Prediction file does not exist")
            continue

        try:
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            if gt_mask is None:
                print(f"Cannot read ground truth file: {gt_path}")
                continue
            if pred_mask is None:
                print(f"Cannot read prediction file: {pred_path}")
                continue

            gt_binary = np.where(gt_mask > 128, 1, 0)
            pred_binary = np.where(pred_mask > 128, 1, 0)

            if gt_binary.shape != pred_binary.shape:
                pred_binary = cv2.resize(pred_binary,
                                         (gt_binary.shape[1], gt_binary.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)

            metrics = calculate_metrics_multiclass(gt_binary, pred_binary)

            gt_bool = (gt_binary == 1).astype(bool)
            pred_bool = (pred_binary == 1).astype(bool)

            has_gt_foreground = np.any(gt_bool)
            has_pred_foreground = np.any(pred_bool)

            hd95_val = np.nan
            assd_val = np.nan
            if has_gt_foreground and has_pred_foreground:
                try:
                    hd95_val = hd95(pred_bool, gt_bool)
                    assd_val = assd(pred_bool, gt_bool)
                except Exception as e:
                    print(f"Error calculating distance metrics ({gt_file}): {e}")

            result_row = {
                'File': gt_file,
                'HD95': hd95_val,
                'ASSD': assd_val
            }
            result_row.update(metrics)

            results.append(result_row)

        except Exception as e:
            print(f"Error processing {gt_file}: {e}")

    if results:
        df = pd.DataFrame(results)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        avg_row = df[numeric_cols].mean()
        avg_row['File'] = 'Average'
        avg_df = pd.DataFrame([avg_row])
        final_df = pd.concat([df, avg_df], ignore_index=True)
        column_order = [
            'File',
            # IoU related metrics
            # 'Background_IoU',
            'Class1_IoU', 'mIoU',
            # DSC related metrics
            # 'Background_DSC',
            'Class1_DSC', 'mDSC',
            # Recall related metrics
            # 'Background_Recall',
            'Class1_Recall', 'mRecall',
            # New: FPR related metrics
            # 'Background_FPR',
            'Class1_FPR', 'mFPR',
            # Distance metrics
            'HD95', 'ASSD'
        ]
        column_order = [col for col in column_order if col in final_df.columns]
        final_df = final_df[column_order]

        # Save results
        os.makedirs(save_folder, exist_ok=True)
        final_df.to_csv(f'{save_folder}/{os.path.basename(pred_folder)}.csv', index=False, float_format='%.4f')
        print(f'Saved {save_folder}/{os.path.basename(pred_folder)}.csv')

        print(f"\nAverage metrics results {os.path.basename(pred_folder)}:")
        print("=" * 70)
        print(f"{'Metric':<25} {'Average':<10} {'Description'}")
        print("-" * 70)

        metric_groups = [
            ('IoU Metrics', [
                ('Background_IoU', 'Background IoU'),
                ('Class1_IoU', 'Target class IoU'),
                ('mIoU', 'Mean IoU')
            ]),
            ('DSC Metrics', [
                ('Background_DSC', 'Background DSC'),
                ('Class1_DSC', 'Target class DSC'),
                ('mDSC', 'Mean DSC')
            ]),
            ('Recall Metrics', [
                ('Background_Recall', 'Background Recall'),
                ('Class1_Recall', 'Target class Recall'),
                ('mRecall', 'Mean Recall')
            ]),
            ('FPR Metrics', [
                ('Background_FPR', 'Background false positive rate'),
                ('Class1_FPR', 'Target class false positive rate'),
                ('mFPR', 'Mean false positive rate')
            ]),
            ('Distance Metrics', [
                ('HD95', '95% Hausdorff Distance (pixels)'),
                ('ASSD', 'Average Symmetric Surface Distance (pixels)')
            ])
        ]

        for group_name, metrics in metric_groups:
            print(f"\n{group_name}:")
            for metric, description in metrics:
                if metric in avg_row and not pd.isna(avg_row[metric]):
                    value = avg_row[metric]
                    print(f"  {metric:<22} {value:<10.4f} {description}")

        print(f"\nVerification calculations:")
        print(
            f"mIoU verification: ({avg_row['Background_IoU']:.4f} + {avg_row['Class1_IoU']:.4f}) / 2 = {avg_row['mIoU']:.4f}")
        print(
            f"mDSC verification: ({avg_row['Background_DSC']:.4f} + {avg_row['Class1_DSC']:.4f}) / 2 = {avg_row['mDSC']:.4f}")
        print(
            f"mRecall verification: ({avg_row['Background_Recall']:.4f} + {avg_row['Class1_Recall']:.4f}) / 2 = {avg_row['mRecall']:.4f}")
        print(
            f"mFPR verification: ({avg_row['Background_FPR']:.4f} + {avg_row['Class1_FPR']:.4f}) / 2 = {avg_row['mFPR']:.4f}")

    return results

