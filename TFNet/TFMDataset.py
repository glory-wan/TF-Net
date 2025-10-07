from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# Data Loading (Adjust according to your dataset structure)
# Assume your data directory structure is as follows:
# data_dir/
# ├── images/
# │    ├── train
# │    │    ├── train1.png
# │    │    ├── train2.png
# │    │    └── ...
# │    ├── val
# │    │    ├── val1.png
# │    │    ├── val2.png
# │    │    └── ...
# │    └── test
# │         ├── test1.png
# │         ├── test2.png
# │         └── ...
# └── masks/ (The structure is the same as that of images)
#     ├── mask1.png
#     ...

Extension = (
    ".bmp", ".dng", ".jpeg", ".jpg", ".mpo",
    ".png", ".tif", ".tiff", ".webp", ".pfm"
)


class TFMDataset(Dataset):
    def __init__(self, root_dir, split=None, imgz=512):
        self.split = split
        self.images_dir = os.path.join(root_dir, 'images', self.split)
        self.masks_dir = os.path.join(root_dir, 'masks', self.split)

        self.images = [os.path.join(self.images_dir, f) for f in
                       tqdm(os.listdir(self.images_dir), desc=f'loading images in {str(self.split)} dataset ')
                       if f.lower().endswith(Extension)]

        self.mask_paths = [os.path.join(self.masks_dir, f) for f in
                           tqdm(os.listdir(self.masks_dir), desc=f'loading masks in {str(self.split)} dataset ')
                           if f.lower().endswith(Extension)]

        original_image_count = len(self.images)
        indices_to_remove = []
        for i, image_path in enumerate(self.images):
            image_name = os.path.splitext(os.path.basename(image_path))[0]  # e.g. tran1
            label_path = os.path.join(self.masks_dir, image_name + '.png')
            if not os.path.exists(label_path):
                indices_to_remove.append(i)
                continue
        for index in sorted(indices_to_remove, reverse=True):
            del self.images[index]
        new_image_count = len(self.images)
        if original_image_count != new_image_count:
            print(f"number of original images in {self.split}: {original_image_count}, \n"
                  f"number of valid images in {self.split}: {new_image_count}")

        self.transform = transforms.Compose([
            transforms.Resize((imgz, imgz)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((imgz, imgz)),
            transforms.ToTensor()
        ])

        if not self.images:
            raise ValueError(f"No images found in directory {self.images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image_name = os.path.splitext(os.path.basename(self.images[idx]))[0]  # e.g. train1
        mask_path = os.path.join(self.masks_dir, image_name + '.png')
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).long()

        return image, mask


def load_seg_datasets(data_dir, batch_size=2, workers=4, imgz=512,
                      train_split='train', val_split='val'):
    train_ = TFMDataset(
        root_dir=data_dir,
        split=train_split,
        imgz=imgz,
    )
    val_ = TFMDataset(
        root_dir=data_dir,
        split=val_split,
        imgz=imgz,
    )

    dataloaders = {
        'train': DataLoader(train_,
                            batch_size=min(batch_size, 2),
                            shuffle=True,
                            pin_memory=True,
                            persistent_workers=True,
                            num_workers=workers,
                            ),
        str(val_split): DataLoader(val_,
                                   batch_size=min(batch_size, 3),
                                   shuffle=False,
                                   pin_memory=True,
                                   persistent_workers=True,
                                   num_workers=workers,
                                   # drop_last=True,
                                   )
    }

    return dataloaders


