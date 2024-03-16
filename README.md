# Assignment 02

This project is an implementation of the SphereFace: Deep Hypersphere Embedding for Face Recognition.
It utilizes a novel A-Softmax loss that enables CNNs to learn angularly discriminative features, making it particularly effective for face recognition tasks.

SphereFace uses A-Softmax Loss, trains and evaluates on LFW dataset.

We will be using 4-layer CNN for this assignment

Default using GPU 0.

## Dataset Preparation
Download the LFW dataset and organize it in the following structure

├── contents
│   ├── lfw  # Unzipped LFW dataset
│   ├── pairsDevTest.txt
│   └── pairsDevTrain.txt

Align and crop images. This can be done using MTCNN.

## Training
run the below code

python main.py --train_file /content/pairsDevTrain.txt --eval_file /content/pairsDevTest.txt --img_folder /content/lfw

## Evaluation
The evaluation process compares the model's predictions against the true labels to determine its performance. Used accuracy as evaluation metric.

## Result
This implementation gets 74.60% on LFW dataset.
