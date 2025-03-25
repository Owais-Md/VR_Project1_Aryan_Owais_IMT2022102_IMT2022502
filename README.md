# Face Mask Detection, Classification, and Segmentation Project

 
## Introduction
------------
This project focuses on developing a computer vision solution to classify and segment face masks in images, addressing a critical need for automated detection systems in contexts like public health monitoring. The objective is to implement and compare two approaches:

- **Binary Classification** : Determining whether a person in an image is wearing a face mask ("with mask") or not ("without mask") using:
  - Handcrafted features with traditional machine learning (ML) classifiers.
  - A Convolutional Neural Network (CNN).
- **Mask Segmentation** : Identifying and delineating the mask region in images of people wearing masks using:
  - Traditional region-based segmentation techniques.
  - A U-Net deep learning model.

The implementation leverages Python, utilizing libraries such as OpenCV, scikit-learn, TensorFlow, and PyTorch, to process images, train models, and evaluate performance.

## Submission
----------
- **Contributors** :
  - Aryan Mishra(IMT2022502, Aryan.Mishra@iiitb.ac.in)
  - Md Owais(IMT2022102, Mohammad.Owais@iiitb.ac.in)
- **GitHub Repository** : https://github.com/Owais-Md/VR_Project1_Aryan_Owais_IMT2022102_IMT2022502
- Files:
  - **BinaryClassification.ipynb** : Tasks A and B.
    - **all_cnn_hyperparameters.csv** : All the hyperparameters used and their results in Task B.
    - **best_cnn_hyperparameters.csv** : The hyperparameters which gave the best results in Task B.
  - **Segmentation.ipynb** : Tasks C and D.
  - **README.md** : This file.

## Dataset
-------
### Sources

The project utilizes two publicly available datasets:

#### Face Mask Detection Dataset:
- **Source** : https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
- **Description** : Contains images of people with and without face masks, labeled for binary classification tasks.
- **Structure** :
```
  dataset
├── with_mask # contains images with mask
└── without_mask # contains images without face-mask
```
- **Access** : Stored in a zip file (finaldataset.zip) with images in JPG format.
        
#### Masked Face Segmentation Dataset (MFSD):
- **Source** : https://github.com/sadjadrz/MFSD
- **Description** : Provides images with corresponding ground truth segmentation masks for faces with masks.
- **Structure** :
```
  MSFD
├── 1
│   ├── face_crop # face-cropped images of images in MSFD/1/img
│   ├── face_crop_segmentation # ground truth of segmend face-mask
│   └── img
└── 2
    └── img
```
- **Access** : Stored in a zip file (MSFD.zip) with 9,382 valid image-mask pairs filtered from the original set.

## Preprocessing
-------------
- **Classification Dataset** : Images are resized to 64x64 pixels, normalized to [0, 1], and split into training (80%) and validation (20%) sets.
- **Segmentation Dataset** : Images and masks are resized to 128x128 pixels, normalized, and split into training (80%) and validation (20%) sets for U-Net training.

## Methodology
-----------
### Task A: Binary Classification Using Handcrafted Features and ML Classifiers
-------------------------------------------------------------------------------
#### A.i: Extract Handcrafted Features

- **Features** : Both Histogram of Oriented Gradients (HOG) features and Scale Invariant Feature Transform(SIFT) features are extracted from the Face Mask Detection dataset and as HOG gave better results, that has been used to train and evaluate the ML classifiers.
- **Process** : Images are loaded from finaldataset.zip, resized to 64x64, and converted to grayscale before both HOG and SIFT feature extraction.

#### A.ii: Train and Evaluate ML Classifiers

- **Classifiers** :
  - **Support Vector Machine (SVM)** : We tried altering between various kernels and finally used 'rbf' kernel as it gave the best results.
  - **Neural Network** : To get the best results, we had to alter in various ways like adding dropout layers, using adam optimiser, experimenting with the number of hidden layers and the number of nodes in each layer to get the best result possible. The final structure is as shown below:
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense_34 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │         <span style="color: #00af00; text-decoration-color: #00af00">451,840</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_24 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_35 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │          <span style="color: #00af00; text-decoration-color: #00af00">32,896</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_25 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_36 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │           <span style="color: #00af00; text-decoration-color: #00af00">8,256</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_26 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_37 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │              <span style="color: #00af00; text-decoration-color: #00af00">65</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>


- **XGBoost** : To get the best results, we create DMatrices(optimised data structure for XGBoost) using the test and train data, experimented between the parameters and used xgb.cv(k fold cross-validation) with early stopping. We got 86 rounds as the best number of rounds for cross validation and used the following parameters: 

   ![Screenshot 2025-03-25 at 4 43 48 PM](https://github.com/user-attachments/assets/7608228b-a2e0-4ff9-96a3-766eb3b8bbc7)


   ![Screenshot 2025-03-25 at 4 44 33 PM](https://github.com/user-attachments/assets/9231294f-5641-431d-b7d3-4162e8c66b34)
   
- **Training**: Features are split into training and validation sets (80-20 split), and classifiers are trained using sklearn.
- **Evaluation**: Accuracy is computed on the validation set using accuracy_score.

#### A.iii: Report and Compare Accuracy

- Results are compared between SVM, Neural Network and XGBoost based on validation accuracy.

### Task B: Binary Classification Using CNN
--------------------------------------------------
#### B.i: Design and Train a CNN

- **Architecture** : We used 3 convolutional blocks followed by fully connected layers followed by the output layer for binary classification(Model: 'sequential_11'). The structure can be seen below:
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">62</span>, <span style="color: #00af00; text-decoration-color: #00af00">62</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">896</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">31</span>, <span style="color: #00af00; text-decoration-color: #00af00">31</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">29</span>, <span style="color: #00af00; text-decoration-color: #00af00">29</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │          <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)          │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">12</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4608</span>)                │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_38 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │         <span style="color: #00af00; text-decoration-color: #00af00">589,952</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_27 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_39 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">129</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>

- **Training** : Images are loaded via a custom ZipDataGenerator, trained with Adam optimizer and binary cross-entropy loss.

#### B.ii: Hyperparameter Variations

- **Variations Tested** :
  - Learning rates: 0.01, 0.001, 0.0001.
  - Optimizers: Adam, SGD, RMSprop.
  - Batch sizes: 16, 32, 64.
  - Dropout rates: 0.3, 0.4, 0.5.
  - Final activation: Sigmoid, ReLU, Softmax.
- **Process**: Models are trained for 5 epochs with early stopping (patience=2) and evaluated on validation accuracy.
- (The hyperparameters used and their results are shown in the section below.)

#### B.iii: Compare CNN with ML Classifiers

- The best CNN configuration’s accuracy is compared against the traditional ML classifiers(SVM, Neural Network and XGBoost) results.

### Task C: Region Segmentation Using Traditional Techniques
--------------------------------------------------------------------
#### C.i: Implement Region-Based Segmentation

- **Method** : K-Means clustering followed by morphological operations.
  - Steps:
    - Load images from MSFD.zip (face_crop).
    - Apply K-Means (k=3) to cluster pixels.
    - Identify mask cluster and refine with binary closing and flood fill.
    - Generate binary mask.

#### C.ii: Visualize and Evaluate

- **Visualization** : Input image, ground truth, and predicted mask are plotted.
- **Metrics** : Intersection over Union (IoU) and Dice score are computed against ground truth masks from face_crop_segmentation.

### Task D: Mask Segmentation Using U-Net
-----------------------------------------------
#### D.i: Train a U-Net Model

- **Architecture** :
  - Encoder: 4 downsampling blocks (Conv2D + ReLU + MaxPooling).
  - Bottleneck: Conv2D block.
  - Decoder: 4 upsampling blocks (UpSampling2D + Conv2D + ReLU) with skip connections.
  - Output: 1x1 Conv2D with Sigmoid activation.
- **Training** : Uses MFSD dataset, resized to 128x128, trained with Adam optimizer and binary cross-entropy loss for 10 epochs.

#### D.ii: Compare U-Net with Traditional Method

- **Evaluation** : IoU and Dice scores are computed on validation set and compared with traditional segmentation results.

## Hyperparameters and Experiments
-------------------------------
### CNN (Task B)
- We tried a total of 12 different hyperparameters experiments by using different learning rates, optimizer, batch size, dropout rates and final activation functions.
- All the hyperparameters combinations along with their results are stored in **all_cnn_hyperparameters.csv**.
- The best hyperparameters combination along with its result is stored in **best_cnn_hyperparameters.csv**.
- **All hyperparameter configurations used** :
 <img width="989" alt="Screenshot 2025-03-25 at 5 30 12 PM" src="https://github.com/user-attachments/assets/6f0e1880-dc14-4169-a9b9-b13222f88748" />

- **Best hyperparameter onfiguration** :
 <img width="989" alt="Screenshot 2025-03-25 at 5 31 43 PM" src="https://github.com/user-attachments/assets/c7f084ba-5746-40bf-a3d9-2d9b8288edf4" />

### U-Net (Task D)

- **Hyperparameters** :
  - Learning rate: 0.001
  - Optimizer: Adam
  - Batch size: 16
  - Epochs: 10
  - Loss: Binary cross-entropy
- **Experiments** : Single configuration trained due to computational constraints, with early stopping considered but not implemented in the provided code.

## Results
-------
### Task A: Traditional ML Classifiers(SVM, Neural Network, XGBoost)

- **SVM** : 93.53% accuracy
- **Neural Network (MLP)** : 91.09% accuracy
- **XGBoost** : 92.43% accuracy
  ![Screenshot 2025-03-25 at 5 36 30 PM](https://github.com/user-attachments/assets/11271674-60b6-4348-8849-c77ab6c050d6)


### Task B: Convolutional Neural Network(CNN)

- **CNN** : 95.85% accuracy {'learning_rate': 0.001, 'optimizer': 'adam', 'batch_size': 32, 'final_activation': 'sigmoid', 'dropout_rate': 0.3}
  ![Screenshot 2025-03-25 at 5 38 12 PM](https://github.com/user-attachments/assets/c7628180-8aa7-44f8-9404-522a25c48bfc)


### Task C: Traditional Segmentation

- **Sample Results (on one image)**:
  - IoU: 0.65
  - Dice: 0.79

### Task D: U-Net Segmentation

- **Validation Results**:
  - Average IoU: 0.82
  - Average Dice: 0.90

## Comparison

- **Classification** : CNN (95.85%) outperforms SVM (93.53%), MLP (91.09%), and XGBoost (92.43%).
- **Segmentation** : U-Net (IoU=0.82, Dice=0.90) significantly outperforms traditional method (IoU=0.65, Dice=0.79).

## Observations and Analysis
-------------------------
### Classification:
- CNN outperforms the traditional ML classifiers used as expected as traditional ML classifiers rely on handcrafted features (like HOG, SIFT), which might not capture all the nuances of the images. CNNs learn hierarchical features that can capture more complex patterns, potentially leading to higher accuracy.
- Hyperparameter tuning (e.g., lower dropout, optimal learning rate) significantly boosts CNN performance.
- **Challenges**: Limited dataset size may lead to overfitting; mitigated by dropout and early stopping.
### Segmentation:
- U-Net provides precise mask delineation thanks to skip connections preserving spatial details, unlike the coarser traditional method.
- Traditional segmentation struggles with complex mask shapes and lighting variations, leading to lower IoU and Dice scores.
- **Challenges**: U-Net training is computationally intensive as the number of channels here are high; traditional methods are faster but less accurate.
### General:
- Deep learning models (CNN, U-Net) consistently outperform traditional approaches, justifying their use despite higher resource demands.
- Dataset quality (e.g., alignment of image-mask pairs) impacts segmentation performance.

## How to Run the Code
-------------------
### Prerequisites
- Python: 3.7+
- Libraries: Install via pip:
```
bash

pip install numpy opencv-python matplotlib scikit-learn scikit-image tensorflow torch torchvision
```
- **Datasets**:
  - Download dataset.zip from GitHub.
  - Download MSFD.zip from GitHub.
  - Place both in the datasets/ directory.

### Directory Structure
-------------------
```
VR_Project1_[YourName]_[YourRollNo]/
├── datasets/
│   ├── finaldataset.zip
│   ├── MSFD.zip
├── classification_notebook.ipynb
├── segmentation_notebook.ipynb
├── README.md
```

### Running Classification (BinaryClassification.ipynb)
--------------------------------------------------------
- Open Notebook:
  ```
    bash
    jupyter notebook BinaryClassification.ipynb
  ```
- Update Paths:
        Set zip_file_path to "datasets/finaldataset.zip".
- Execute Cells:
  - Run all cells sequentially to:
    - Load and preprocess data.
    - Train ML classifiers (Task A).
    - Train and tune CNN (Task B).
- Outputs: Accuracy metrics for SVM, MLP, and CNN.

### Running Segmentation (Segmentation.ipynb)
--------------------------------------------------
- Open Notebook:
  ```
    bash
    jupyter notebook Segmentation.ipynb
  ```
- Update Paths:
        Set zip_file_path to "datasets/MSFD.zip".
- Execute Cells:
        Run all cells to:
            Perform traditional segmentation (Task C).
            Train and evaluate U-Net (Task D).
- Outputs: Visualizations, IoU, and Dice scores.

## Notes
-----
- Ensure sufficient RAM and GPU (if available) for U-Net training.
- Outputs are printed in the notebook; no additional intervention is required.
