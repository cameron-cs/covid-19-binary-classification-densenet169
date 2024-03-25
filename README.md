# covid-19-binary-classification-densenet169

## COVID-19 binary classification with a custom DenseNet169 model

This project aims to leverage deep learning for the binary classification of CT scan images to detect COVID-19 infection. Using a customized version of the DenseNet169 architecture, we tackle the challenge of distinguishing between COVID-19 positive and negative cases, contributing valuable tools for assisting healthcare professionals in diagnosis.

## Dataset

The model is trained and evaluated using the SARS-CoV-2 CT scan dataset, which is publicly available on Kaggle. This dataset comprises CT scan images categorized into two classes: COVID and non-COVID. It serves as a valuable resource for developing machine learning models to assist in diagnosing COVID-19.

## Data source
The dataset can be downloaded from the following link: [SARS-CoV-2 CT scan dataset](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset).

### Dataset characteristics

- **Total images**: the dataset contains a total of 2482 CT scan images.
- **Classes**: there are two classes of images:
  - `COVID`: CT scans of patients infected with SARS-CoV-2.
  - `non-COVID`: CT scans of patients without SARS-CoV-2 infection.
- **Image format**: the images are stored in PNG format.
- **Resolution**: images vary in size. For training purposes, images are resized to a uniform resolution of 224x224 pixels as part of the preprocessing steps.

### Usage

Before training the model, ensure you download and extract the dataset into a directory accessible to the training script. The `data_dir` configuration parameter should be set to the path of this directory.

### Acknowledgments

We extend our gratitude to the dataset contributors for making this valuable resource available to the research community, thereby facilitating advancements in medical imaging and machine learning for public health.


## Overview

The project encompasses data preprocessing, model training with a custom DenseNet169 model, and evaluation on a test dataset. The key components include:

- `preprocessor.py`: Handles image loading, preprocessing (resizing and normalization), and dataset splitting.
- `dataset.py`: A custom data generator extending `keras.utils.Sequence` for efficient data handling during model training.
- `custom_densenet169.py`: Defines the CustomDenseNet model, adapting the DenseNet169 architecture for our binary classification task.
- `training.py`: Orchestrates model training, including data augmentation, compiling the model, setting callbacks, and saving the best model.

### Model Explanation

The CustomDenseNet model extends the pre-trained DenseNet169 architecture, known for its efficiency and performance in image classification tasks. The custom model is defined as follows:

The core of the Custom DenseNet169 model can be represented as follows:

$$
\text{Output} = \text{Softmax}\Big(\text{Dense}_{64} \big(\text{BN}(\text{GAP}(\text{DenseNet169}_{\text{base}}(X)))\big)\Big)
$$

Where:
- \(X\) is the input image.
- \(\text{DenseNet169}_{\text{base}}\) denotes the base DenseNet169 model pre-trained on ImageNet, excluding its top classification layer.
- \(\text{GAP}\) stands for Global Average Pooling, reducing spatial dimensions to a single vector.
- \(\text{BN}\) represents Batch Normalization, stabilizing the learning process.
- \(\text{Dense}_{64}\) is a densely connected layer with 64 units followed by a ReLU activation.
- \(\text{Softmax}\) outputs the probabilities of the two classes, indicating the likelihood of COVID-19 infection.

### Project setup

#### Requirements

Ensure you have Python 3.6+ installed along with the following packages:
- TensorFlow 2.x
- Keras
- OpenCV
- scikit-learn
- NumPy

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Model performance

After training the Custom DenseNet169 model on the COVID-19 CT scan dataset, we evaluated its classification performance on a separate test set. The model demonstrated high precision and recall across both categories (COVID and non-COVID), as summarized in the classification report below:

- **Precision**: the model's precision scores of 0.97 for COVID and 0.95 for non-COVID indicate a high level of reliability in its positive classifications. In other words, when the model predicts an image as COVID or non-COVID, it is correct 97% and 95% of the time, respectively.

- **Recall**: the recall scores of 0.94 for COVID and 0.97 for non-COVID suggest that the model is also highly sensitive to detecting positive cases. It correctly identifies 94% of all actual COVID cases and 97% of all actual non-COVID cases.

- **F1-score**: the F1-scores, which balance precision and recall, are both high (0.96 for both categories), indicating robust overall performance.

- **Accuracy**: with an overall accuracy of 96%, the model demonstrates strong capability in distinguishing between COVID-19 positive and negative cases from CT scans.

These results underscore the potential of using deep learning models like the Custom DenseNet169 for aiding in the rapid and accurate diagnosis of COVID-19, thereby supporting healthcare systems in effective disease management.