# coding:utf-8
#kerasでGrad-CAM 自分で作ったモデルで
#https://qiita.com/haru1977/items/45269d790a0ad62604b3


import pandas as pd
import numpy as np
import cv2
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model

K.set_learning_phase(1) #set learning phase

def Grad_Cam(input_model, x, layer_name):
    '''
    Args:
       input_model: モデルオブジェクト
       x: 画像(array)
       layer_name: 畳み込み層の名前

    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)

    '''

    # 前処理
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0


    # 予測クラスの算出

    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]


    #  勾配を取得

    conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成

    cam = cv2.resize(cam, (200, 200), cv2.INTER_LINEAR) # 画像サイズは200で処理したので
    cam = np.maximum(cam, 0) 
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成

    return jetcam

def Grad_Cam_plus_plus(input_model, layer_name, x, row, col):

    model = input_model

    # 前処理
    X = np.expand_dims(x, axis=0)
    X = X.astype('float32')
    preprocessed_input = X / 255.0


    # 予測クラスの算出
    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])

    #  使用する重みの抽出、高階微分の計算
    class_output = model.layers[-1].output
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(class_output, conv_output)[0]
    #first_derivative：１階微分
    first_derivative = K.exp(class_output)[0][class_idx] * grads
    #second_derivative：２階微分
    second_derivative = K.exp(class_output)[0][class_idx] * grads * grads
    #third_derivative：３階微分
    third_derivative = K.exp(class_output)[0][class_idx] * grads * grads * grads

    #関数の定義
    gradient_function = K.function([model.input], [conv_output, first_derivative, second_derivative, third_derivative])  # model.inputを入力すると、conv_outputとgradsを出力する関数


    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = gradient_function([preprocessed_input])
    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = conv_output[0], conv_first_grad[0], conv_second_grad[0], conv_third_grad[0]

    #alphaを求める
    global_sum = np.sum(conv_output.reshape((-1, conv_first_grad.shape[2])), axis=0)
    alpha_num = conv_second_grad
    alpha_denom = conv_second_grad*2.0 + conv_third_grad*global_sum.reshape((1,1,conv_first_grad.shape[2]))
    alpha_denom = np.where(alpha_denom!=0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom

    #alphaの正規化
    alpha_normalization_constant = np.sum(np.sum(alphas, axis = 0), axis = 0)
    alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))
    alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad.shape[2]))

    #wの計算
    weights = np.maximum(conv_first_grad, 0.0)
    deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad.shape[2])))

    #Lの計算
    grad_CAM_map = np.sum(deep_linearization_weights * conv_output, axis=2)
    grad_CAM_map = np.maximum(grad_CAM_map, 0)
    grad_CAM_map = grad_CAM_map / np.max(grad_CAM_map)

    #ヒートマップを描く
    grad_CAM_map = cv2.resize(grad_CAM_map, (row, col), cv2.INTER_LINEAR)
    jetcam = cv2.applyColorMap(np.uint8(255 * grad_CAM_map), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成

    return jetcam

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

img = cv2.imread('3.jpg')
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_resize = cv2.resize(img, (int(200), int(200)), interpolation=cv2.INTER_CUBIC) 
#x = img_to_array(load_img('3.jpg', target_size=(28,28)))
x = img_to_array(img_resize)

array_to_img(x)

image = Grad_Cam(model, x, 'conv2d_3') 
array_to_img(image)
plt.imshow(array_to_img(image))
plt.show()


