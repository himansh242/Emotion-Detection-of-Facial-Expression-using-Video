from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
# number of channels
img_channels = 1
path1 = 'trainimages'    #path of folder of images    
path2 = 'new_dataset_resize_for_classification' 
#path3 =  r'C:\Users\hp\Desktop\video\validation'
#path of folder to save images    
img_rows, img_cols = 224, 224
listing = os.listdir(path1) 

num_samples=size(listing)
print (num_samples)

for file1 in listing:
    print(path1 +'/' + file1)
    im = Image.open(path1 +'/' + file1)   
    img = im.resize((img_rows,img_cols))
    #gray = img.convert('L')
                #need to do some more processing here           
    img.save(path2 +'/' +  file1)

imlist = os.listdir(path2)

im1 = array(Image.open(path2 + '/'+ imlist[0])) # open one image to get size
m,n,o = im1.shape[0:3] # get the size of the images
imnbr = len(imlist) # get the number of images
print(o)
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten() for im2 in imlist],'f')

label=np.ones((num_samples,),dtype = int)
label[0:348]=0
label[348:637]=1

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]
print (train_data[1].shape)

img=immatrix[167].reshape(img_rows,img_cols,3)
print (train_data[0].shape)

batch_size = 8
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 20


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
(X, y) = (train_data[0],train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
import keras
# convert class vectors to binary class matrices
Y_train =keras.utils.np_utils.to_categorical(y_train, nb_classes)
Y_test = keras.utils.np_utils.to_categorical(y_test, nb_classes)

i = 100



def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = shuffled_data[start_index: end_index], shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()



 
import keras
vgg16_model = keras.applications.vgg16.VGG16()
# load the model
model = VGG16()
print(model.summary())
type(vgg16_model)
models= Sequential()
for layer in vgg16_model.layers:
    models.add(layer)
models.summary()
models.layers.pop()
for layer in models.layers[:5]:
    layer.trainable = False
models.summary()
   
models.add(Dense(2,activation = 'sigmoid'))
models.summary()
#models.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
models.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
models.fit(X_train, Y_train,epochs =1) 
for layer in models.layers[:10]:
    layer.trainable = False
for layer in models.layers[10:]:
    layer.trainable = True
print(X_train.shape)
train_steps, train_batches = batch_iter(X_train, Y_train, batch_size)
valid_steps, valid_batches = batch_iter(X_test, Y_test, batch_size)
models.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy']) 
models.summary()
#models.fit_generator(train_batches, 509,validation_data=valid_batches,validation_steps=valid_steps, epochs=2, verbose=1)
# load an image from file
model.load_weights('Classifier_weights.h5')
from sklearn.metrics import log_loss
predictions_valid = models.predict(X_test, batch_size=batch_size, verbose=1)
print(predictions_valid)
arr1 = np.argmax(predictions_valid, axis=1)
print(arr1)
arr1 = np.argmax(Y_test, axis=1)
print(arr1)

