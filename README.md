# Plant-Classification

Additional criteria that I worked on to get the best score:

Implement Callback
The images in the dataset have non-uniform resolution.
The dataset used contains more than 10,000 images.
The accuracy of the training set and validation set is at least 95%.
Has 3 or more classes.
Perform inference using one of the models (TF-Lite, TFJS or savedmodel with tf serving).
Project Explanation
This project is a project to create a model that can classify images. Given the freedom to choose the dataset you want to use.

Dataset
The dataset is data taken from GitHub. The dataset has a total of 14 plants divided into 38 different classes. By default, the resolution of the image is 256x256, but to meet the criteria, the dataset is randomly changed to a size with a minimum range of 200x200 to 256x256.

Preview Image
Due to hardware limitations for training, only tomato plants were selected. Here are examples of images from each class of tomato plants:

<img width="558" alt="image" src="https://github.com/user-attachments/assets/d07b4d57-b87c-42af-8f37-7134c19d10cf" />


Image Distribution
From 10 tomato classes, 4 classes were selected again with the distribution of each class as follows:

Condition Number of Images
<img width="302" alt="image" src="https://github.com/user-attachments/assets/6b4db9b4-736e-4e08-a06a-1b69458e4698" />


Evaluation Model
Model Architecture
MobileNetV2 Pre-trained:

Using MobileNetV2 that has been trained on ImageNet by removing the top layer (include_top=False).
The model input size is (150, 150, 3).
Frozen Layer:

All MobileNetV2 layers are frozen (layer.trainable = False) to maintain the trained weights and features.
Additional Layers:

Conv2D layer with 32 filters, kernel size 3x3, and ReLU activation, followed by MaxPooling2D with pool size 2x2.
Conv2D layer with 64 filters, kernel size 3x3, and ReLU activation, followed by MaxPooling2D with pool size 2x2.
Flatten and Fully Connected Layers:

The features obtained from the previous layer are flattened using Flatten.
Dropout layer with rate 0.5 to prevent overfitting.
Dense layer with 128 units and ReLU activation.
Dense output layer with 4 units and softmax activation for multi-class classification.

Accuracy and Loss Graph
<img width="495" alt="image" src="https://github.com/user-attachments/assets/e9786ea2-ede1-42a8-a1d5-2cb8430e43e5" />



Predict
No True Predicted
1 Healthy Healthy
2 Late_blight Late_blight
3 Septoria_leaf_spot Septoria_leaf_spot
4 Tomato_Yellow_Leaf_Curl_Virus Tomato_Yellow_Leaf_Curl_Virus
How To Inference
Inference Using TensorFlow Serving.

