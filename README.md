# Bird Classification using EfficientNet

## Overview
This project focuses on classifying birds into 400 distinct classes. The dataset comprises 58,388 training images, 2,000 test images, and 2,000 validation images. We leverage the EfficientNet B0 architecture from PyTorch's torchvision library and fine-tune it on our dataset to achieve impressive accuracy.

## Dataloader
The dataset undergoes specific augmentations for the training set to ensure the model is robust to various transformations such as flipping, color variations, blurring, and erasing. On the other hand, the test and validation datasets undergo standard transformations, mainly resizing and normalization.

The dataloader code provided allows users to fetch data in batches and applies necessary transformations to ensure it's ready for model training.

## Model Architecture
We utilize the EfficientNet B1 architecture, a state-of-the-art model known for its efficient performance. We've set all its layers to non-trainable (except the classifier). We replace the final classifier layer to match our dataset's number of classes (400).

Here is a glimpse of our new classifier:
(nn.Linear(n_inputs, 2048),
nn.SiLU(),
nn.Dropout(0.2),
nn.Linear(2048, len(classes)))


## Training & Evaluation
The model is trained using the Cross-Entropy Loss with label smoothing. The AdamW optimizer is employed with a learning rate of `0.001`.

The training function `train_model` fine-tunes the model over a specified number of epochs. For each epoch, it calculates loss and accuracy for both training and validation phases. Additionally, it keeps track of the best weights (in terms of validation accuracy) during the entire training process.

Post-training, the model's performance is evaluated on the test dataset using the `test` function, providing a detailed class-wise accuracy report and an overall test accuracy.

## Results
- **Test Accuracy (Overall):** 97.3790%
- **Best Validation Accuracy:** 96.250%

## Usage
To utilize the notebook for your dataset:

1. Ensure your data directory structure matches:
   
├── data_dir
├── train
├── valid
├── test

2. Adjust the batch size, learning rate, or number of epochs as per your requirements.
3. Run the cells in the notebook to train and evaluate the model.

## Dependencies
Make sure you have the following libraries:
- PyTorch
- torchvision
- numpy

## Contribute
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

