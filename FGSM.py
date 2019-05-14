#ディープラーニングを騙す
#https://qiita.com/ryuoujisinta/items/2c566ebea4bc43a62632

from __future__ import print_function
import keras
from keras.datasets import cifar10,cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Reshape, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam, SGD

import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
#from keras.utils.visualize_util import 
import cv2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.vgg19 import VGG19 #, preprocess_input, decode_predictions
from keras.applications.xception import Xception #, preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50 #, preprocess_input, decode_predictions
from keras.applications.densenet import DenseNet121 #, preprocess_input, decode_predictions
from keras.applications.inception_resnet_v2 import InceptionResNetV2 #, preprocess_input, decode_predictions
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.backend import variable
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from keras.datasets import cifar10
import keras


model = VGG16(weights='imagenet')
#model1.summary()

"""
model = VGG19(weights='imagenet')
#model.summary()

model3 = Xception(weights='imagenet')  #add size=(299,299)
#model3.summary()

model4 = ResNet50(weights='imagenet') #add
#model4.summary()

model5 = DenseNet121(weights='imagenet') #cancat
#model5.summary()

model6 = InceptionResNetV2(weights='imagenet') #ac size=(299,299)
#model6.summary()
"""

size=(224,224)
frame = cv2.imread("cat_dog.png")
preprocessed_input= cv2.resize(frame, size)
preprocessed_input= np.expand_dims(preprocessed_input, axis=0)
predictions = model.predict(preprocessed_input)
print('Predicted class:')
for s in range(5):
    top_1 = decode_predictions(predictions)[0][s]
    print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

    
def fast_gradient(model, img, eps):
    #target = variable(img)
    target=img
    predict_result = model.predict(target)
    orig_ind = np.argmax(predict_result)
    print(orig_ind)
    # create adv array
    loss = K.sum(K.square(model.output - target))
    grads = K.gradients(loss, model.input)[0]
    fn = K.function([model.input, target], [K.sign(grads)])
    
    adv_part = np.sign(target.grad)
    adv_array = target.data + eps * adv_part
    return adv_array.astype(np.float32), adv_part, predict_result
    
#FGSM.py
def FGSM(model, img,y_test):
    target = K.placeholder()
    loss = K.sum(K.square(model.output - target))
    grads = K.gradients(loss, model.input)[0]
    fn = K.function([model.input, target], [K.sign(grads)])
    grad_sign = []
    for i in range(20):
        part = np.arange(i*500 , (i + 1) * 500)
        grad_sign.append(fn([img[part], y_test[part]])[0])
        #print(fn([img[part], y_test[part]]))
    grad_sign = np.concatenate(grad_sign)
    
    eps = 0 #1e-5 #0.25
    eps_img=eps * grad_sign
    x_adv = np.clip(img + eps * grad_sign, 0, 1)
    print(x_adv[0].shape)
    return x_adv,eps_img

# データをロード
img_rows, img_cols, ch=32,32,3
num_classes = 10
# データをロード
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# サイズ変更
X_train =[]
X_test = []
for i in range(10000):
    dst = cv2.resize(x_train[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    X_train.append(dst)
for i in range(10000):
    dst = cv2.resize(x_test[i], (img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    X_test.append(dst)

X_train = np.array(X_train).reshape(10000,img_rows, img_cols,ch)
X_test = np.array(X_test).reshape(10000,img_rows, img_cols,ch)    
y_train=y_train[:10000]
y_test=y_test[:10000]
x_train = X_train.astype('float32')
x_test = X_test.astype('float32')
x_train /= 255
x_test /= 255    
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

input_tensor = Input(shape=x_train.shape[1:])  #(img_rows, img_cols, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:])) #vgg16,vgg19,InceptionV3,ResNet50,MobileNet,DenseNet121
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=top_model(vgg16.output))


img=preprocessed_input
eps=1e-5
x_adv, eps_img=FGSM(model,x_test, y_test)
#adv_array, adv_part, predict_result=fast_gradient(model, img, eps)
plt.imshow(array_to_img(x_adv[0].reshape(img_rows, img_cols,ch)))
plt.show()
plt.imshow(array_to_img(eps_img[0].reshape(img_rows, img_cols,ch)))
plt.show()

import numpy as np
from sklearn.metrics import confusion_matrix

#check x_test
#predictions = model.predict(X_test, verbose=1)
predict_classes = np.argmax(model.predict(x_test[:100,]),1)
print(predict_classes)
true_classes = np.argmax(y_test[:100],1)
print(true_classes)
print(confusion_matrix(true_classes, predict_classes))