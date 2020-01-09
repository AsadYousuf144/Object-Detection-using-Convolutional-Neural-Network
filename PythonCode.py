from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np

# loading the data, split between train and test sets (each consisting of corresponding image and label arrays)
(images_train, labels_train), (images_test, labels_test) = fashion_mnist.load_data()

#########################################################################
# Optional visualisation tools to help you understand the data.         #
# You can modify and this section (or ignore it) as you wish.           #
# The visualisation will no longer work after the data is preprocessed. #
# None of this will impact  points or grading of this task!             #
#########################################################################
import matplotlib.pyplot as plt

#creating an array with class names in correct position
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print the shape of the training data
print('images_train shape:', images_train.shape)

#print the amount of training and test samples
print(images_train.shape[0], 'train samples')
print(images_test.shape[0], 'test samples')

#how visualize a single image
print('\n colored example visualization of an image:')


# plt.figure()
# plt.imshow(images_train[7])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#how the data of a single image is stored in the dataset
print('\n example array structure of an image (28 arrays with 28 entries each):\n', images_train[7] )

#how the data of a single label is stored in the dataset (one single value between 0-10 representing the class)
label = labels_train[7]
print('\n example label of an image:', label)
print(' this corresponds to class:', class_names[label])

#how to visualize the first 15 images with class name
#this can be used to verfiy that the data is still in correct format e.g. after transforming it


print('\n example visualization of 15 images with class name:')
plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels_train[i]])

#plt.show()

################################################
# Preprocessing.
################################################

# define number of classes
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    images_train = images_train.reshape(images_train.shape[0], 1, img_rows, img_cols)
    images_test = images_test.reshape(images_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    images_train = images_train.reshape(images_train.shape[0], img_rows, img_cols, 1)
    images_test = images_test.reshape(images_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# scale pixel values (ranging from 0 to 255) to range of 0 to 1
# this is a normal pre-processing step for efficiency reasons
images_train = images_train.astype('float32')
images_test = images_test.astype('float32')
images_train /= 255
images_test /= 255


# convert class vectors to binary class matrices
# e.g. 6 becomes [0,0,0,0,0,1,0,0,0,0]
# this step is required to train your model with the categorical cross entropy loss function
# a function especially suited for multiclass classification tasks
labels_train = keras.utils.to_categorical(labels_train, num_classes)
labels_test = keras.utils.to_categorical(labels_test, num_classes)


print("Before Train Shapes: ", images_train.shape)
print("Before Test Shapes: ", images_test.shape)

# print("Here is the labels Train data")
# print(labels_train[0:10])

# print("Here is the labels Test data")
# print(labels_test[0:10])

#create sequential model
model = keras.Sequential()

############################### IMPLEMENTATION OF CONVOLUTIONAL NEURAL NETWORK #############################

## We will use sequential model which allows us to build layer by layer model.
## Used Add functions to add layers to our model


##################################### CONVOLUTIONAL 2D LAYERS ############################################# 
## These convolutional layers will deal with our input images 
## 64 and 32 are the no of nodes in layer with 3x3 and 4x4 Filter matrix.
## Rectified Linear for activation function

model.add(Conv2D(32,3,data_format='channels_last', activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64,4,data_format='channels_last', activation='relu',input_shape=(28,28,1)))

############################################################################################################
##################################### MAX POOLING LAYER ####################################################
model.add(MaxPooling2D(pool_size=(2,2)))
############################################################################################################
## Dropout drops out 30% of weights in the learning process
model.add(Dropout(0.3))

## Now we will Flatten the array to get the Fully Connected Network and let the next layer have 256 Nodes
## by issuing Dense funtion
## Flatten basically converts all out matrix to single vector

model.add(Flatten())

model.add(Dense(256))
############################################################################################################
## Now we again drops the 50% of the weights as mentioned in the Model Architecture
## and then we add fully connected output layer with 10 nodes because we have 10 classes 
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

## Now we compile it 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

####################################### TRAINING THE MODEL #################################################
## To train our model we will use FIT funtion with parameters as (Training Data, Target Data, Validation data, Epochs )
#### For validation we will use provided test set and the no of Epochs is the number of times the model will cycle through the data. 

model.fit(images_train,labels_train, validation_data=(images_test,labels_test), epochs=3, batch_size=100)

####################################### PREDICT ON SINGLE IMAGE ############################################
# reshape the data 
pred_model = model.predict(images_test[100].reshape(1,28,28,1))

## We use np.argmax to turn into actual digits. Printing the actual class name 

print("\n Our model predicted -> ",np.argmax(pred_model, axis=1)) 

## Actual results to compare our prediction model from test set.
print("\n Actual Results to Compare our model -> ",labels_test[100])

####   MODEL PREDICTION ->  [3]
####   ACTUAL RESULT ->  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]

### By comparing that our model predicted 3 for the image at position [100] and the actual result from test set 
### also shows 3 for the image at position [100]. So our model predicted correctly. 