import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

from TFNet import reparameterize_model, create_tfnet_variants
from TFNet.TFMDataset import load_seg_datasets


def predict_single_batch(model, images, device, num_classes):
    """对单个batch进行预测"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)

        if num_classes == 2:
            if outputs.shape[1] == 1:
                preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).long()
            else:
                preds = torch.argmax(outputs, dim=1)
        else:
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy().astype(np.uint8)


def simple_segmentation_predict(model, input_folder, output_folder, device, imgz=512, split='test'):
    model.eval()
    model = reparameterize_model(model)
    model.inference_mode = True

    os.makedirs(output_folder, exist_ok=True)

    segloaders = load_seg_datasets(input_folder, val_split=split, batch_size=1, workers=4, imgz=imgz)
    dataloader = segloaders[split]
    pbar = tqdm(dataloader, desc=f'Predicting {split} masks')
    image_count = 0

    image_filenames = []
    for img_path in dataloader.dataset.images:
        if isinstance(img_path, str):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            image_filenames.append(base_name)
        else:
            image_filenames.append(None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 2:
                images, masks = batch
            else:
                images = batch[0]

            pred_masks = predict_single_batch(model, images, device, num_classes=2)

            for i in range(pred_masks.shape[0]):
                pred_mask = pred_masks[i]

                # 确保掩码是2D的 (H, W)
                if pred_mask.ndim == 3:
                    # 如果是3D，取第一个通道或者压缩维度
                    if pred_mask.shape[0] == 1:
                        pred_mask = pred_mask[0]  # (1, H, W) -> (H, W)
                    else:
                        pred_mask = pred_mask.squeeze()

                if pred_mask.ndim != 2:
                    print(f"Warning: Unexpected mask shape {pred_mask.shape}, squeezing...")
                    pred_mask = pred_mask.squeeze()

                visual_mask = np.zeros_like(pred_mask, dtype=np.uint8)
                visual_mask[pred_mask == 1] = 255

                if visual_mask.ndim != 2:
                    print(f"Warning: visual_mask has unexpected shape {visual_mask.shape}, squeezing...")
                    visual_mask = visual_mask.squeeze()

                base_name = image_filenames[image_count]
                mask_path = os.path.join(output_folder, f"{base_name}.png")

                try:
                    mask_img = Image.fromarray(visual_mask)
                    mask_img.save(mask_path)
                    # print(f"✓ Saved mask: {mask_filename} (shape: {visual_mask.shape})")
                except Exception as e:
                    print(f"✗ Error saving mask {mask_path}: {e}")
                    print(
                        f"Mask info - shape: {visual_mask.shape}, dtype: {visual_mask.dtype}, range: [{visual_mask.min()}, {visual_mask.max()}]")
                    continue

                image_count += 1

            pbar.set_description(f'Predicting {split}: {image_count}/{len(dataloader.dataset)}')


def predict_binary_mask(
        model_path, image_dir, output_path,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("Note: Used weights_only=False for compatibility")

    models_ = create_tfnet_variants(num_classes=checkpoint['nc'], width=checkpoint['wid'])
    model = models_['full']

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)

    simple_segmentation_predict(
        model=model,
        input_folder=image_dir,
        output_folder=output_path,
        device=device
    )
