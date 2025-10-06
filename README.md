# TFM Dataset & Benchmark: Automated Tear Film Break-Up Analysis 

**📄 Paper:** *"TFM Dataset: A Novel Multi-task Dataset and Integrated Pipeline for Automated Tear Film Break-Up Segmentation"* 

**🏠 Project Page:** *[coming soon]*

This repository is the official implementation of our IEEE TMI paper. We introduce the **Tear Film Multi-task (TFM) Dataset**, the first comprehensive public benchmark for multi-task tear film analysis, alongside **TF-Net**, an efficient segmentation model, and **TF-Collab**, an integrated pipeline for fully automated tear film break-up (TFBU) analysis.

## 🚀 News

- **[2025-10-07]**  🎉 TFM Dataset of TF-Net is released!

## ✨ Highlights

- **🧬 TFM Dataset:** The first public dataset with **6,247 high-resolution images** annotated for three vision tasks: **frame classification**, **Placido ring detection**, and **pixel-wise TFBU segmentation**.
- **⚡ TF-Net:** A novel, computationally efficient segmentation model using a **re-parameterized MobileOne-mini backbone** and **Pyramid Pooling Module**, achieving real-time performance.
- **🔄 TF-Collab Pipeline:** An integrated, end-to-end pipeline that synergistically combines classification, detection, and segmentation models to fully automate TFBU analysis, mirroring the clinical workflow.
- **🏆 Benchmark:** We provide benchmark results against numerous state-of-the-art medical image segmentation models.

## 📊 TFM Dataset Overview

The TFM Dataset is designed to foster research in automated dry eye diagnosis.

### Dataset Statistics



### Annotation Examples

Overview of the **Tear Film Multi-task (TFM)** Dataset composition, illustrating the distribution and relationships between the three annotation tasks: classification (TF-Cls), object detection (TF-Det), and segmentation (TF-Seg).

![dataset](assets/dataset.jpg)

Sample visualization of the **TF-Crop** dataset from cropping strategy.. The first row displays the original full-resolution images (left) and their corresponding cropped versions (right), which are generated based on the "Outside" bounding boxes from the TF-Det dataset. The second row presents the visualizations of the pixel-wise TFBU segmentation masks for the respective images above.

![crop](assets/crop.jpg)

### Download

The TFM Dataset can be downloaded from the following sources:



**📝 License for Data:** The TFM Dataset is available for **non-commercial research purposes only**. Please see the  file for details. By downloading, you agree to the terms.

## 🛠️ Installation & Quick Start

### 1. Environment Setup

```
git clone https://github.com/glory-wan/TF-Net.git
cd TF-Net

conda create -n TFNet python=3.11 -y
conda activate TFNet

pip install -r requirements.txt
```

### 2. Data Preparation

1. Download the TFM Dataset from the link above.
2. Extract it into the `dataset/` directory. The structure should look like:

```
TFM dataset/
    ├── images/
    │    ├── train
    │    │    ├── train1.png
    │    │    ├── train2.png
    │    │    └── ...
    │    ├── val
    │    │    ├── val1.png
    │    │    ├── val2.png
    │    │    └── ...
    │    └─── test
    │         ├── test1.png
    │         ├── test2.png
    │         └── ...
    ├── labels/
    │    ├── classification
    │    │    ├── train_cls.txt
    │    │    ├── val_cls.txt
    │    │    └── test_cls.txt
    │    ├── detection
    │    │    ├── train
    │    │    │     ├── train1.txt
    │    │    │     └── ...
    │    │    │
    │    │    ├── val
    │    │    │     ├── val1.txt
    │    │    │     └── ...
    │    │    └─── test
    │    │          ├── test1.txt
    │    │    	    └── ...
    │    └─── segmentaiton (The structure is the same as detection)
    │         └── ...
    └── TF-Crop/ # The pre-cropped ROI subset for segmentation
```

TF-Crop(即crop_broken_only，用于分割模型训练)，crop-full，crop-non-broken用于消融实验。

The structure of TF-Crop should look like:

```
 crop_broken_only/
      (crop_full)/
(crop_non_broken)/
        ├── images/
        │    ├── train
        │    │    ├── train1.png
        │    │    ├── train2.png
        │    │    └── ...
        │    ├── val
        │    │    ├── val1.png
        │    │    ├── val2.png
        │    │    └── ...
        │    └─── test
        │         ├── test1.png
        │         ├── test2.png
        │         └── ...
        └── masks/ (The structure is the same as that of images)
            ├── ...
            ...
```

## 🧠 Model Zoo & Benchmark

### TF-Net & trained Models

We provide trained weights for TF-Net and other baseline models.



### Training & Evaluation

1. **Train TF-Net from scratch:**

   ```
   
   ```

2. **Predict on the test set:**

   ```
   
   ```

3. **calculate metrics**: Iou, DSC, Recall, FPR, HD95, ASSD

   ```
   
   ```

## 📈 Results

### Quantitative Segmentation Results on TF-Crop Test Set

Our TF-Net achieves a favorable balance between accuracy and speed, making it ideal for real-time clinical applications.

![Compare_result](assets/Compare_result.jpg)

## 📄 License

This project's code is released under the . The TFM Dataset is for **non-commercial research use only**. Please refer to  for the dataset's license terms.