import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


#Tensorflow librairies
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, Input, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from sklearn import metrics
from tensorflow.keras.applications import MobileNetV2

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2) #output = 32 x 224 x 224
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding = 2) # output = 64 x 224 x 224
        self.pool = nn.MaxPool2d(2,2) #output = 64 x 112 x 112
        self.conv2_drop = nn.Dropout2d(p=0.3)

        #We do a 2 maxpooling before using these layers to reduce the dimension of the features from 224x224 to 56x56
        self.fc1 = nn.Linear(64*56*56, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x)) # first layer: conv + relu
        x = self.pool(x) # 2nd layer : pooling
        x = torch.relu(self.conv2(x)) # 3rd layer: conv + relu
        x = self.pool(x) # 4th layer : pooling
        x = self.conv2_drop(x) #th layer : dropout
        x = x.view(-1, 64*56*56)
        x = torch.relu(self.fc1(x)) # 6th layer : FC + relu
        x = self.fc2(x) # 7th layer : FC
        return x



#tensorflow model
def get_tensorflow_model():
    #We load the pretrained model ResNet50
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3)) #96x96x3 input shape

    #we freeze the layers
    for layer in base_model.layers:
        layer.trainable = False
    
    #We add new layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x) #To flatten the output
    x = Dense(56, activation='relu')(x) #FC
    x = Dropout(0.3)(x)
    predictions = Dense(4, activation='softmax')(x) #4 classes

    model = Model(inputs=base_model.input, outputs=predictions)

    return model