# Enhanced Image Classification with Deep Learning and ESRGAN

## Overview
This repository is centered around an image classification project that explores the impact of image resolution on classification accuracy. Using the CIFAR-10 dataset, we apply a two-stage deep learning process: first, enhancing image resolution using the Enhanced Super-Resolution Generative Adversarial Network (ESRGAN), and then classifying these images into 10 distinct categories using a Convolutional Neural Network (CNN).

## Project Components
- **ESRGAN for Image Enhancement**: The project utilizes ESRGAN to upscale low-resolution CIFAR-10 images, resulting in higher resolution versions for improved classification.
- **CNN for Image Classification**: A CNN is trained to classify both the original low-resolution and the enhanced high-resolution images to evaluate performance differences.
- **Comparative Analysis**: The classification results are compared to determine if high-resolution images lead to increased accuracy in identifying the correct class labels.

## Setup and Environment
The project requires Python with specific libraries like TensorFlow, Keras, and others which are crucial for running ESRGAN and CNN models.

## Dataset
The CIFAR-10 dataset, consisting of 32x32 pixel color images across 10 classes, serves as the foundation for this study.

## Usage
Instructions on how to utilize the pre-trained ESRGAN model to enhance image resolution, followed by steps to train and evaluate the CNN classifier, are included in the repository.

## Repository Contents

- `Image_Processing/`: Scripts for image upscaling and preprocessing.
- `CNN_Classification/`: The neural network model scripts for classification tasks.
- `Results_Analysis/`: Documentation on experiment results and analytical discussions.

## Results
The repository presents an in-depth analysis demonstrating that classifiers trained on higher resolution images tend to yield higher accuracy, validating the hypothesis that image resolution can significantly impact classification performance.

![image](https://github.com/khullarsanket/Image-Classification-using-Deep-Learning-to-Enhance-the-Classification-accuracy-using-ESRGAN/assets/119709438/0573ccd7-4916-4909-a805-4eb5f7ac1f97)


## Conclusion
The study concludes that super-resolution can be a valuable pre-processing step in image classification tasks, potentially leading to more accurate results in practical applications.

## References
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ESRGAN GitHub Repository](https://github.com/xinntao/ESRGAN)

