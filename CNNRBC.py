## Set up
import numpy as np
import keras
from keras import backend as K
from keras.model import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
%matplotlib inline

train_path = 'data/train'
test_path = 'data/test'

## Data processing c'est chaud 

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(90,90),classes=['slippers', 'croissants','sheared'], batch_sizes=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(90,90),classes=['slippers', 'croissants','sheared'], batch_sizes=10)
#valid_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(90,90),classes=['slippers', 'croissants','sheared'], batch_sizes=10)


## Model 
## 3 couches de convolutions
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(90,90,1))
cnn_model.add(LeakyReLU(alpha=0.25))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu')
cnn_model.add(LeakyReLU(alpha=0.25))
cnn_model.add(MaxPooling2D(pool_size=(2, 2),))
cnn_model.add(Conv2D(128, (3, 3), activation='relu')
cnn_model.add(LeakyReLU(alpha=0.25))                  
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
## On écrase les matrices en vecteurs
cnn_model.add(Flatten())
## Couche fully-connected
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(LeakyReLU(alpha=0.25))                  
cnn_model.add(Dense(num_classes, activation='softmax'))

## Compile the model

cnn_model.compile(
    loss=keras.losses.categorical_crossentropy, 
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

## On regarde la tête de notre modèle 
cnn_model.summary()

## Training 
epochs = 10 # nombre donné pour le modèle de PLOS 

## Il faut un jeu de validation 

cnn_train = cnn_model.fit_generator(train_batches, 
    step_per_epochs=5 
    epochs=epochs,
    verbose=1,
    validation_data=valid_batches
)