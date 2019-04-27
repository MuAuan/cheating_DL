# coding:utf-8

import numpy as  np
import matplotlib.pyplot as plt
import tensorflow as tf
#import tensorflow.keras.backend as K
from keras import backend as K
import keras
from keras.datasets import mnist
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint

from keras.layers import *
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import cv2
from keras.datasets import cifar10,cifar100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train,y_train,x_test,y_test = getDataSet(img_rows,img_cols)
img_rows, img_cols=200,200

X_train =[]
X_test = []
for i in range(50000):
    dst = cv2.resize(x_train[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC) #cv2.INTER_LINEAR #cv2.INTER_CUBIC
    dst = dst[:,:,::-1]  
    X_train.append(dst)
for i in range(10000):
    dst = cv2.resize(x_test[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    dst = dst[:,:,::-1]  
    X_test.append(dst)
X_train = np.array(X_train)
X_test = np.array(X_test)

y_train=y_train[:50000]
y_test=y_test[:10000]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

x_train = X_train.astype('float32')
x_test = X_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#def model_mnist(input_image=Input(shape=(None, None, 1))):
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',input_shape=(200,200,3)))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=3))  
model.add(Dropout(0.5))                
model.add(AveragePooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(BatchNormalization(axis=3))  
model.add(Dropout(0.5))                
model.add(AveragePooling2D((2, 2)))    
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))  
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))  
#x = BatchNormalization(axis=3)(x)  
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))
model.summary()

#model = model_mnist(input_image=Input(shape=(28, 28, 1)))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])

#model = load_model("mnist_cnn_adv.hdf5")
#model.load_weights('mnist_cnn.hdf5')

checkpointer = ModelCheckpoint(filepath='./cifar10/cifar_cnn.hdf5', 
                               monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max',
                               verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                               factor=0.5, min_lr=0.00001, verbose=1)
csv_logger = CSVLogger('./cifar100/history_cifar_cnn.log', separator=',', append=True)
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

#Learning ; Original x_train, y_train
history = model.fit(x_train, y_train,
          batch_size=64,
          epochs=20,
          callbacks=callbacks,          
          validation_split=0.2,
          shuffle=True) 
          
          
