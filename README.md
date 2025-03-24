# Mask Detection and Segmentation Project

## 1. Introduction
This project focuses on detecting and segmenting face masks in images using various machine learning and deep learning techniques. The tasks include binary classification using handcrafted features and machine learning classifiers, binary classification using CNNs, region segmentation using traditional techniques, and mask segmentation using U-Net. The objective is to evaluate the performance of different approaches in terms of classification accuracy and segmentation quality.

### Contributors:

(IMT2022502) Aryan Mishra <Aryan.Mishra@iiitb.ac.in>

(IMT2022102) Md Owais <Mohammad.Owais@iiitb.ac.in>

---

## 2. Dataset
### Source:
https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset

https://github.com/sadjadrz/MFSD

```
MSFD
├── 1
│   ├── face_crop # face-cropped images of images in MSFD/1/img
│   ├── face_crop_segmentation # ground truth of segmend face-mask
│   └── img
└── 2
    └── img
```

```
dataset
├── with_mask # contains images with mask
└── without_mask # contains images without face-mask
```

### Structure:
- **Training Set:** Images used for training the models.
- **Testing Set:** Images used to evaluate model performance.
- **Annotations:** Mask region labels for segmentation tasks.

---

## 3. Objectives

### a. Binary Classification Using Handcrafted Features and ML Classifiers
1. Extract handcrafted features from facial images (e.g., HOG, LBP, SIFT).
2. Train and evaluate at least two machine learning classifiers (Here we use XGBoost and Neural Network) and compare classifier performances based on accuracy.

### b. Binary Classification Using CNN
1. Design and train a Convolutional Neural Network (CNN) for mask classification.
2. Experiment with different hyperparameters (learning rate, batch size, optimizer, activation function).
3. Compare CNN performance with traditional ML classifiers.

### c. Region Segmentation Using Traditional Techniques
1. Apply region-based segmentation methods (e.g., thresholding, edge detection) to segment mask regions, visualize and evaluate segmentation results.

### d. Mask Segmentation Using U-Net
1. Train a U-Net model to segment the mask regions in facial images.
2. Compare segmentation performance with traditional techniques using IoU or Dice score.

---

## 4. Hyperparameters and Experiments
 [TO BE ADDED BY DAKSH AND ADITYA]

---

## 5. Results
### Evaluation Metrics:
- **Classification:** Accuracy, Precision, Recall, F1-score
- **Segmentation:** Intersection over Union (IoU), Dice Score

| Model | Accuracy (%) | IoU | Dice Score |
|--------|------------|----|-----------|
| XGBoost (part a) | 94.15% (80-20 train-test split) | - | - |
| Neural Network (part a)| 91.25% (80-20 train-test split) | - | - |
| CNN | [TO BE ADDED BY DAKSH AND ADITYA] |
| Region-growing (part c) | - | 0.3559 (mean) | 0.4798 (mean) |
| K-mean clustering  (part c) | Explained in section 6|
| U-Net Segmentation | [TO BE ADDED BY DAKSH AND ADITYA] |

---

## 6. Observations and Analysis

### Part a

For each image here we need to make a feature vector. We choose 5 features: color features, HoG, Edge features, texture featuresand ORB fetaures. 

***Since feature vector coresponding to images may be of diffrent lentgh, we resize all image and fix the length of individual sub-feature vectors, so that `np.hstack() `can work without interrupts when all individual sub-feature vectors ar combined into one vector for an image***. Data used is `dataset`. We train an XGBoost model as well as a neural network and as observed, the test accuracy of XGBoost is better. This is attributed to the fact that neural networks need a lot of data to learn and here we have 4095 images.

### Part b

[TO BE ADDED BY DAKSH AND ADITYA]

### Part c

2 techniques used: K-means clustering based segmentation and Region-growing.

For K-means, k=2, one for mask region and another for backround.

Here for the choice of the 2 initial centroids, we use domain knowledge. The images are cropped to face-size which implies that it is higly likely that some region of the mask must be in the center of image. 

So we choose one centoid at center and another at corner.

For Region-based segmentaion, choice of initial seed here(only one) is center of the image, the 'why' of it backed by the reasoning provided above.

We find that K-means captures all pixels as part of mask that have more or less the intensity as mask. Secondly it is found that if the masks have design patterns of high contast, they are inevitably left out in mask segment no matter how much blurring you apply.
<p align="center">
  <img src="images/3_1.png" width="65%" />
  <img src="images/3_1_gt.jpg" width="25%" />
</p>
[Results from K-means and ground truth mask for `MSFD/1/000003.jpg`]


We find that Region-growing technique is ***sensitive to tolerance***(the threshold difference for pixels be considered connected to seed). In cases where the tone of skin is comparable to that of face-mask, the tolerance needs to be drastically low to capture correct pixels.

<p align="center">
  <img src="images/58_1_rg.png" width="65%" />
  <img src="images/58_1_gt.jpg" width="25%" />
</p>
[Results from region-growing and ground truth mask for `MSFD/1/000058_1.jpg`]

<p align="center">
  <img src="images/7_1_rg.png" width="45%" />
  <img src="images/7_1.png" width="45%" />
</p>

[Results from region-growing and k-means for `MSFD/1/000007_1.jpg`. Less false-positives in Region-growing.]

<p align="center">
  <img src="images/13_1_rg.png" width="45%" />
  <img src="images/13_1.png" width="45%" />
</p>

[Results from region-growing and k-means for `MSFD/1/000013_1.jpg`. Less false-positives in Region-growing.]

In conclusion:
| K-means | Region-growing |
|--------|------------|
| Slower| Relatively faster|
| Highly likely to give false-positives (cases where mask tone matches hair, spectacle,etc)| Less likely to give false positives|
| Sensitive to number of iterations| Sensitive to tolerance|

Both the algorihtms rely on predefined parameters, they do not 'learn' and hence fail to generalise over large dataset (poor mean IoU and Dice scores). Computing mean IoU and Dice for K-means over 8500+ images is computationally expensive, moreover it is evident from its performance over random samples that its scores won't be significantly better region-growing.




---


### PART D

# Project Report: Image Segmentation using Traditional and Deep Learning Methods

## i. Introduction
This project focuses on implementing image segmentation techniques using both traditional region-based methods and deep learning models such as CNN and U-Net. The objective is to segment facial regions accurately and compare the effectiveness of different methodologies.

## ii. Dataset
- **Source**: The dataset used consists of cropped facial images with corresponding ground truth masks.
- **Structure**:
  - `face_crop/`: Contains input images.
  - `face_crop_segmented/`: Contains ground truth segmentation masks.
  - `output/`: Stores results from segmentation techniques.

## iii. Methodology
### **Traditional Segmentation (Part C)**
- **Thresholding and Morphological Operations**: Basic segmentation based on pixel intensity.
- **Region-Based Segmentation**: Methods such as flooding and binary closing were applied.
- **K-Means Clustering**: Used to segment regions based on color similarity.

### **Deep Learning Models (Part D)**
- **CNN-based Segmentation**: Trained on facial images to predict masks.
- **U-Net Architecture**: A powerful fully convolutional network trained for pixel-wise classification.

## iv. Hyperparameters and Experiments
- **CNN Model**:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Batch Size: 32
  - Number of Epochs: 50
  - Loss Function: Categorical Crossentropy

- **U-Net Model**:
  - Optimizer: Adam
  - Learning Rate: 0.0001
  - Batch Size: 16
  - Number of Epochs: 100
  - Loss Function: Dice Loss

Different variations of learning rates, optimizers, and batch sizes were tested to fine-tune the models.

## v. Results
- **Evaluation Metrics**:
  - Accuracy
  - Intersection over Union (IoU)
  - Dice Similarity Score

  <p align="center">
  <img src="images/unetpics.png" width="45%" />
  
</p>


| Model | Accuracy | IoU | Dice Score |
|--------|------------|------|------------|
| U-Net | 0.9664 | 0.9137 | 0.9509 |

As we can see the unet model works much better than traditional methods

<p align="center">
  <img src="images/unetresults" width="45%" />
  
</p>


## vi. Observations and Analysis
- **Traditional methods** work well for simple segmentation tasks but struggle with complex images.
- **CNN-based models** improve segmentation but may require extensive data augmentation.
- **U-Net** outperforms other approaches, providing the highest accuracy and IoU.
- Challenges include dealing with varying lighting conditions and occlusions, which were addressed using preprocessing techniques and data augmentation.

## 7. How to Run the Code
### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/JConquers/VR_Project_1
   cd VR_PROJECT_1
   ```
2. Install dependencies:
    
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
3. Download the dataet from the source specified and put the 2 repositores `dataset` and `MSFD` at same directory level, immediately below repository level. Make directory `output`, command `mkdir output`. Final structure must look like :
    ```
    .
    ├── dataset
    ├── MSFD
    ├── output
    ├── scripts
    └── images
    
    # Other files like README.md, pdf, etc are not shown in this tree.
    ```
4. Run the scripts:
   
   `\scripts` contains 2 notebooks `part_a_b.ipynb` and `part_c_d.ipynb`, which contains scripts for the respective parts. They can be run all at once or one at a time to see partial results.

---

## 8. Conclusion
This project demonstrates the effectiveness of deep learning techniques for face mask detection and segmentation. CNN models outperform traditional classifiers for binary classification, while U-Net provides more precise segmentation results. Further improvements can be achieved by using more complex architectures and larger datasets.

---

--------------------------------------------------------------------------------------------------------
