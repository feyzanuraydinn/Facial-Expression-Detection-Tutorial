# Real-Time Facial Expression Detection Tutorial

## Tutorial Overview

This tutorial will guide you through the process of creating a real-time facial expression detection system using Python, Keras, and OpenCV. Facial expression detection has various applications, from user experience enhancement to market research. By the end of this tutorial, you will have a working system that can analyze live video streams and predict the emotional state of individuals.

## Installation

You'll need a working installation of Python to run the code examples and scripts. The code in this tutorial is written in Python 3.11.4.
Install required libraries using pip:

```
pip install -r requirements.txt
```
## Dataset

You can download the dataset used in this tutorial from [here](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset?select=images). Alternatively, you can proceed with the tutorial by creating your own custom dataset and working with your own data.

## Data Format and File System

Before you start training your model, make sure that your dataset's file format is suitable. Just like this:

<details open><summary>data</summary><blockquote>
        <details open><summary>train<summary><blockquote>
        <details open><summary>angry<summary><blockquote>
        image_1.jpg <br>
        image_2.jpg <br>
        image_3.jpg <br>
        </blockquote></details>
        <details open><summary>disgust<summary><blockquote>
        image_x.jpg <br>
        image_y.jpg <br>
        image_z.jpg <br>
        </blockquote></details>
        ...
        <details open><summary>validation<summary><blockquote>
        <details open><summary>angry<summary><blockquote>
        image_1.jpg <br>
        image_2.jpg <br>
        image_3.jpg <br>
        </blockquote></details>
        <details open><summary>disgust<summary><blockquote>
        image_x.jpg <br>
        image_y.jpg <br>
        image_z.jpg <br>
        </blockquote></details>
        ...
        </blockquote></details>
</blockquote></details>

The data folder should be in the same directory as your project.

## Train Model

To train your model using the dataset, run 'train.py'.

This code creates and trains a Convolutional Neural Network (CNN) model for recognizing emotional expressions. The dataset consists of emotional facial images located in the 'data/train/' and 'data/test/' directories. The training data is augmented using various techniques such as rotation, shearing, zooming, and horizontal flipping. The model architecture comprises multiple convolutional layers, max-pooling layers, and dropout layers for regularization. The training progress and performance metrics are printed to the console. Finally, the trained model is saved as 'model_file.h5'.

## Test Model

To test the model, run the 'test.py' script.

This Python script utilizes a trained emotion detection model that we have created to perform real-time analysis of facial expressions on live webcam video. After loading our trained model and the required libraries, the script captures frames from the webcam, identifies faces, and predicts facial expressions for each detected face. It outlines the faces with rectangles and displays the predicted facial expressions above them. Moreover, it exhibits facial expression percentages in the top-right corner of the video feed. The script operates in a continuous loop until the user presses the 'q' key, providing an interactive experience to observe real-time facial expression predictions on detected faces.