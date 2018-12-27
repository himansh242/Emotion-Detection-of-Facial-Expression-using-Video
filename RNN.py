import re
import numpy as np
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb

from keras.utils.np_utils import to_categorical

import warnings
warnings.filterwarnings('ignore')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
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
#import matplotlib.pyplot as plt
#import matplotlib
import os
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

img_channels = 1
path1 = 'input_data'    #path of folder of images    
path2 = 'resize' 
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
    img.save(path2 +'/' +  file1, "png")

imlist = os.listdir(path2)

im1 = array(Image.open(path2 + '/'+ imlist[0])) # open one image to get size
m,n,o = im1.shape[0:3] # get the size of the images
imnbr = len(imlist) # get the number of images

immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')

label=np.ones((num_samples,),dtype = int)
label[0:10]=0
label[10:20]=1
label[20:30]=2
label[30:40]=3
label[40:50]=4
label[50:60]=5
label[60:70]=6
#label[61:71]=6



data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]
print (train_data[1].shape)

img=immatrix[40].reshape(img_rows,img_cols,3)
#plt.imshow(img)

#plt.imshow(img,cmap='gray')
print (train_data[0].shape)

batch_size = 32
# number of output classes
nb_classes = 6
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

X_train = X_train.reshape(X_train.shape[0], img_rows*img_cols*3)
X_test = X_test.reshape(X_test.shape[0], img_rows*img_cols*3)

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
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)
X_train[0]
print('Build model...')
model = Sequential()
model.add(Embedding(70, 8))
model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, 
          batch_size=batch_size, 
          epochs=3, 
          validation_data=(X_test, Y_test))
model.save_weights('RNN_weights.h5')