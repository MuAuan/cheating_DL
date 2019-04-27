#ディープラーニングを騙す
#https://qiita.com/ryuoujisinta/items/2c566ebea4bc43a62632

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

def plot_gallery(images, titles, h, w, n_row=9, n_col=9):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(0.9 * n_col, 1.2 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(81):  #n_row*n_col
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = y_pred[i]  #target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[i]   #.rsplit(' ', 1)[-1]
    print(pred_name, true_name)
    return 'predicted: {}\ntrue:  {}'.format(pred_name, true_name)



# データをロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 前処理
x_train = np.expand_dims(x_train, 3)
x_train = x_train.astype("float32") / 255

x_test = np.expand_dims(x_test, 3)
x_test = x_test.astype("float32") / 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
"""
# モデルの定義
# A common Conv2D model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu',padding='same',input_shape=(28,28,1)))
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
"""

#def model_mnist(input_image=Input(shape=(None, None, 1))):
model = Sequential()
model.add(Conv2D(32, (2, 2), activation="relu", padding="same",input_shape=(28,28,1)))
model.add(Conv2D(128, (2, 2), activation="relu", padding="valid"))
model.add(Conv2D(128, (1, 1), activation="relu", padding="valid"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))
model.summary()

#model = model_mnist(input_image=Input(shape=(28, 28, 1)))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])
#model.load_weights('mnist_cnn_cifar.hdf5')

# 訓練

#es = keras.callbacks.EarlyStopping(monitor="val_acc", patience=1)
checkpointer = ModelCheckpoint(filepath='./cifar100/mnist_cnn.hdf5', 
                               monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max',
                               verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                               factor=0.5, min_lr=0.00001, verbose=1)
csv_logger = CSVLogger('./cifar100/history_mnist.log', separator=',', append=True)
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

#Learning ; Original x_train, y_train
history = model.fit(x_train, y_train,
          batch_size=64,
          epochs=2,
          callbacks=callbacks,          
          validation_split=0.2,
          shuffle=True) 


history = model.fit(x_train, y_train, epochs=2,
                    batch_size=64,
                    callbacks=callbacks,
                    validation_split=0.2)

model.save_weights('mnist_cnn.hdf5', True)
#save_history(history, os.path.join("./histry/", 'history_mnist.txt'),0)

#FGSM.py
target = K.placeholder()
loss = K.sum(K.square(model.output - target))
grads = K.gradients(loss, model.input)[0]
fn = K.function([model.input, target], [K.sign(grads)])
grad_sign = []
for i in range(20):
    part = np.arange(i * 500, (i + 1) * 500)
    grad_sign.append(fn([x_test[part], y_test[part]])[0])
grad_sign = np.concatenate(grad_sign)

eps = 0.25
x_adv = np.clip(x_test + eps * grad_sign, 0, 1)
print(x_adv[0].shape)

#x_train[:1000] = x_train[:1000]+ Noise
target = K.placeholder()
loss = K.sum(K.square(model.output - target))
grads = K.gradients(loss, model.input)[0]
fn = K.function([model.input, target], [K.sign(grads)])
grad_sign = []
for i in range(100):  #20
    part = np.arange(i * 500, (i + 1) * 500)
    grad_sign.append(fn([x_train[part], y_train[part]])[0])
grad_sign = np.concatenate(grad_sign)

eps = 0.25
x_train[:50000] = np.clip(x_train[:50000] + eps * grad_sign[:50000], 0, 1)
print(x_train[0].shape)

import numpy as np
from sklearn.metrics import confusion_matrix

#check x_test
#predictions = model.predict(X_test, verbose=1)
predict_classes = model.predict_classes(x_test[:10000,], batch_size=32)
true_classes = np.argmax(y_test[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(20)]
plot_gallery(x_test[:20], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_test.jpg')
plt.pause(1)
plt.close()

#check x_train[:10000] += Noise
predict_classes = model.predict_classes(x_train[:10000,], batch_size=32)
true_classes = np.argmax(y_train[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_train, true_classes, i) for i in range(20)]
plot_gallery(x_train[:20], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_train.jpg')
plt.pause(1)
plt.close()


#check x_adv[:10000] += Noise
predict_classes = model.predict_classes(x_adv[:10000,], batch_size=32)
true_classes = np.argmax(y_test[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(20)]
plot_gallery(x_adv[:20], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_adv.jpg')
plt.pause(1)
plt.close()



# MNISTデータの表示
W = 10  # 横に並べる個数
H = 10   # 縦に並べる個数
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_test
for i in range(W*H):
    ax1 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax1.imshow(x_test[i].reshape((28, 28)), cmap='gray')
plt.show()

fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_train[:10000] += Noise
for i in range(W*H):
    ax1 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax1.imshow(x_train[i].reshape((28, 28)), cmap='gray')
plt.show()

fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_adv = x_test + Noise
for i in range(W*H):
    ax2 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax2.imshow(x_adv[i].reshape((28, 28)), cmap='gray')
plt.show()



""" 
#model_mnist
 (28, 28, 1)
[[ 976    0    0    0    0    0    1    1    1    1]
 [   0 1131    2    1    1    0    0    0    0    0]
 [   4    7 1008    4    2    0    1    5    1    0]
 [   1    0    2  994    0    7    0    4    1    1]
 [   1    3    1    0  967    0    2    0    0    8]
 [   2    0    2    6    0  879    3    0    0    0]
 [   8    2    1    0    2    2  942    0    1    0]
 [   1    4   14    1    0    0    0  996    6    6]
 [   7    2    4    2    1    2    5    7  941    3]
 [   5    7    1    2    5    4    0    6    7  972]]
 
  #check x_train[:50000] += Noise
[[ 95   4 343  14   1  51 193   0 213  87]
 [  0 578 110  22  27  11   6  91 280   2]
 [ 30 114 294 247  20   0   8  71 195  12]
 [ 15  15 176 180   1 313   0  46 193  93]
 [ 19  56  60   3 112   4  31  69 219 407]
 [ 16  11   5 176   3 164  94   0 317  77]
 [118  53  63   7 171 201 357   1  41   2]
 [  8  57 113  71  26   7   0 404  27 357]
 [ 26 110 346  68  12 153  53  20  82  74]
 [ 17  11  22  50 176  44   2 369 252  35]]
 
 #check x_adv = x_test + Noise
[[ 90   1 344   8   3  40 212   2 189  91]
 [  6 636  91  18  16  12   9  55 291   1]
 [ 31 130 321 298  12   0  10  70 152   8]
 [  6   5 184 154   1 364   3  49 161  83]
 [ 19  36  49   0 111   7  29  84 207 440]
 [ 17   5   6 217   0 116  66   2 385  78]
 [143  28  65  13 157 189 336   0  26   1]
 [  5  67 139  59  17   7   0 375  34 325]
 [ 37  55 349  69  15 195  60  50  64  80]
 [ 14  15  14  54 189  72   1 376 223  51]]
"""

model.load_weights('mnist_cnn.hdf5')

#es = keras.callbacks.EarlyStopping(monitor="val_acc", patience=1)
checkpointer = ModelCheckpoint(filepath='./cifar100/mnist_cnn_adv.hdf5', 
                               monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max',
                               verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                               factor=0.5, min_lr=0.00001, verbose=1)
csv_logger = CSVLogger('./cifar100/history_mnist_adv.log', separator=',', append=True)
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

history = model.fit(x_train, y_train,
          batch_size=64,
          epochs=20,
          callbacks=callbacks,          
          validation_split=0.2,
          shuffle=True) 

model.save_weights('mnist_cnn_adv.hdf5', True)

#FGSM.py
target = K.placeholder()
loss = K.sum(K.square(model.output - target))
grads = K.gradients(loss, model.input)[0]
fn = K.function([model.input, target], [K.sign(grads)])
grad_sign = []
for i in range(100):
    part = np.arange(i * 500, (i + 1) * 500)
    grad_sign.append(fn([x_train[part], y_train[part]])[0])
grad_sign = np.concatenate(grad_sign)

eps = 0.25
x_adv_adv = np.clip(x_adv[:10000] + eps * grad_sign[:10000], 0, 1)
print(x_adv_adv[0].shape)

# MNISTデータの表示
W = 10  # 横に並べる個数
H = 10   # 縦に並べる個数
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_test
for i in range(W*H):
    ax1 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax1.imshow(x_test[i].reshape((28, 28)), cmap='gray')
plt.show()
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_train[:50000] += Noise(1st) 
for i in range(W*H):
    ax1 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax1.imshow(x_train[i].reshape((28, 28)), cmap='gray')
plt.show()
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_adv = x_test + Noise(1st)
for i in range(W*H):
    ax2 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax2.imshow(x_adv[i].reshape((28, 28)), cmap='gray')
plt.show()
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_adv_adv = x_adv+ Noise; x_test is original
for i in range(W*H):
    ax3 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax3.imshow(x_adv_adv[i].reshape((28, 28)), cmap='gray')
plt.show()


import numpy as np
from sklearn.metrics import confusion_matrix
#check x_test
#predict_classes = model.predict_classes(x_test[1:10000,], batch_size=32)
predict_classes = model.predict_classes(x_test[:10000,], batch_size=32)
true_classes = np.argmax(y_test[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(20)]
plot_gallery(x_test[:20], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_test_after_Tr.jpg')
plt.pause(1)
plt.close()

#check x_train[:50000] += Noise(1st)
predict_classes = model.predict_classes(x_train[:10000,], batch_size=32)
true_classes = np.argmax(y_train[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(20)]
plot_gallery(x_train[:20], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_train_after_Tr.jpg')
plt.pause(1)
plt.close()

#check x_adv = x_test + Noise(1st)
predict_classes = model.predict_classes(x_adv[:10000,], batch_size=32)
true_classes = np.argmax(y_test[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(20)]
plot_gallery(x_adv[:20], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_adv_after_Tr.jpg')
plt.pause(1)
plt.close()

#check x_adv_adv = x_adv + Noise(2nd)
predict_classes = model.predict_classes(x_adv_adv[:10000,], batch_size=32)
true_classes = np.argmax(y_test[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(20)]
plot_gallery(x_adv_adv[:20], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_adv_adv_after_Tr.jpg')
plt.pause(1)
plt.close()

"""
 #Train[:50000]+ Noise
 #model_mnist
 (28, 28, 1)
 #check x_test
[[ 830    2   13    2    3    0  110    4    5   11]
 [   0 1127    4    0    1    0    1    0    2    0]
 [   5   37  896    8   13    0    3   23   33   14]
 [   6    6   20  839    0   19    0    7   22   91]
 [   2    9    0    0  918    0    2    0    0   51]
 [  18    7    0   18    0  709   21    7   26   86]
 [  19   40    1    0   27    5  861    0    2    3]
 [   5   16   67    5   10    1    0  636    0  287]
 [  13   59   39    3   35   12   23   14  627  149]
 [   5    9    0    1   59    1    0    7    2  925]]
 
 #check x_train[:50000] += Noise(1st)
[[1001    0    0    0    0    0    0    0    0    0]
 [   0 1127    0    0    0    0    0    0    0    0]
 [   0    0  991    0    0    0    0    0    0    0]
 [   0    0    0 1032    0    0    0    0    0    0]
 [   0    0    0    0  980    0    0    0    0    0]
 [   0    0    0    0    0  863    0    0    0    0]
 [   0    0    0    0    0    0 1014    0    0    0]
 [   0    0    0    0    0    0    0 1070    0    0]
 [   0    0    0    0    0    0    0    0  944    0]
 [   0    0    0    0    0    0    0    0    0  978]]
 
 #check x_adv = x_test + Noise(1st)
[[ 978    0    0    0    0    0    0    2    0    0]
 [   0 1132    1    1    0    0    1    0    0    0]
 [   0    1 1027    0    1    0    0    0    2    1]
 [   0    0    0 1010    0    0    0    0    0    0]
 [   0    0    0    0  982    0    0    0    0    0]
 [   0    0    0    1    0  891    0    0    0    0]
 [   0    1    0    0    1    1  955    0    0    0]
 [   1    1    0    4    2    0    0 1019    0    1]
 [   0    0    0    3    0    1    1    0  969    0]
 [   1    0    0    2    2    1    0    1    2 1000]]
 
 #check x_adv_adv = x_adv + Noise(2nd)
[[ 43  34 108 228  78 125  37  62 187  78]
 [ 32  62 115 277  71 154  45  92 206  81]
 [ 39  35 124 240  71 116  39  88 202  78]
 [ 34  29  97 267  70 122  36  76 183  96]
 [ 28  31 100 238 112 113  33  85 155  87]
 [ 22  21  87 213  76 138  31  73 150  81]
 [ 30  32  93 220  72 134  53  74 171  79]
 [ 39  33 107 248  76 118  42  95 177  93]
 [ 50  25 104 215  62 144  43  74 183  74]
 [ 37  34 104 229  77 125  35  88 200  80]]
""" 

#Carlini-Wagner_L2_attack.py
import tensorflow as tf
#from keras.datasets import mnist
import keras
import numpy as np
import random
from PIL import Image

# 著者の実装
from nn_robust_attacks.l2_attack import CarliniL2

# 著者の実装の一部
def generate_data(data, samples, targeted=True, start=0, inception=False):
    
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


class MNIST_data:
    def __init__(self):
        _, (x_test, y_test) = mnist.load_data()
        x_test = np.expand_dims(x_test, 3)
        self.test_data = x_test.astype("float32") / 255
        self.test_labels = keras.utils.to_categorical(y_test)


class MNISTModel:
    def __init__(self):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (2, 2),
                                      activation="relu",
                                      padding="same",
                                      input_shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(128, (2, 2),
                                      activation="relu",
                                      padding="valid"))
        model.add(keras.layers.Conv2D(128, (1, 1),
                                      activation="relu",
                                      padding="valid"))
        model.add(keras.layers.Flatten())
        # softmaxは適用しない
        model.add(keras.layers.Dense(10))

        # 訓練済みモデルのロード
        model.load_weights('mnist_cnn.hdf5')

        self.model = model

    def predict(self, data):
        return self.model(data)


N = 100

model = Sequential()
model.add(Conv2D(32, (2, 2), activation="relu", padding="same",input_shape=(28,28,1)))
model.add(Conv2D(128, (2, 2), activation="relu", padding="valid"))
model.add(Conv2D(128, (1, 1), activation="relu", padding="valid"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))
model.summary()

#model = model_mnist(input_image=Input(shape=(28, 28, 1)))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])
model.load_weights('mnist_cnn.hdf5')

with tf.Session() as sess:
    data, model1 = MNIST_data(), MNISTModel()
    attack = CarliniL2(sess, model1, batch_size=100, max_iterations=1000,
                       confidence=0, boxmin=0, boxmax=1)

    inputs, targets = generate_data(data, samples=N, targeted=True,
                                    start=0, inception=False)
    print(targets)
    adv = attack.attack(inputs, targets)

# MNISTデータの表示
W = 10  # 横に並べる個数
H = 10   # 縦に並べる個数
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
for i in range(W*H):
    ax1 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax1.imshow(x_test[i].reshape((28, 28)), cmap='gray')
plt.savefig('./cifar100/mnist_x_test100.jpg')
plt.pause(1)
plt.close()

fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
for i in range(W*H):
    ax2 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax2.imshow(adv[i].reshape((28, 28)), cmap='gray')
plt.savefig('./cifar100/mnist_adv100.jpg')
plt.pause(1)
plt.close()

import numpy as np
from sklearn.metrics import confusion_matrix

predict_classes = model.predict_classes(x_test[:81,], batch_size=32)
true_classes = np.argmax(y_test[:81],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(x_test[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_L2_attack_x_test81.jpg')
plt.pause(1)
plt.close()

predict_classes = model.predict_classes(adv[:81,], batch_size=32)
true_classes = np.argmax(y_test[:81],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(adv[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_L2_attack_adv81.jpg')
plt.pause(1)
plt.close()

"""

[[ 8  0  0  0  0  0  0  0  0  0]
 [ 0 14  0  0  0  0  0  0  0  0]
 [ 0  0  8  0  0  0  0  0  0  0]
 [ 0  0  0 11  0  0  0  0  0  0]
 [ 0  0  0  0 14  0  0  0  0  0]
 [ 0  0  0  0  0  7  0  0  0  0]
 [ 0  0  0  0  0  0 10  0  0  0]
 [ 0  0  0  0  0  0  0 14  0  0]
 [ 0  0  0  0  0  0  0  0  2  0]
 [ 0  0  0  0  0  1  0  0  0 10]]
[[0 2 1 1 0 1 1 0 2 0]
 [0 1 3 3 0 4 1 1 1 0]
 [1 2 1 1 0 0 1 0 1 1]
 [2 1 0 0 3 1 2 1 0 1]
 [0 1 3 0 4 1 2 3 0 0]
 [1 0 0 0 0 0 2 1 1 2]
 [2 0 1 2 1 1 1 1 0 1]
 [1 1 1 1 0 1 1 2 3 3]
 [0 0 0 1 0 0 0 0 1 0]
 [2 1 0 2 1 1 0 1 2 1]]

[[ 8  0  0  0  0  0  0  0  0  0]
 [ 0 14  0  0  0  0  0  0  0  0]
 [ 0  0  8  0  0  0  0  0  0  0]
 [ 0  0  0 10  0  0  0  0  1  0]
 [ 0  0  0  0 14  0  0  0  0  0]
 [ 0  0  0  0  0  7  0  0  0  0]
 [ 0  0  0  0  0  0 10  0  0  0]
 [ 0  0  0  0  0  0  0 15  0  0]
 [ 0  0  0  0  0  0  0  0  2  0]
 [ 0  0  0  0  0  0  0  0  0 11]]
 
 [[ 0  8  0  0  0  0  0  0  0  0]
 [ 0 14  0  0  0  0  0  0  0  0]
 [ 0  8  0  0  0  0  0  0  0  0]
 [ 0 11  0  0  0  0  0  0  0  0]
 [ 0 14  0  0  0  0  0  0  0  0]
 [ 0  7  0  0  0  0  0  0  0  0]
 [ 0 10  0  0  0  0  0  0  0  0]
 [ 0 15  0  0  0  0  0  0  0  0]
 [ 0  2  0  0  0  0  0  0  0  0]
 [ 0 11  0  0  0  0  0  0  0  0]]
 
 [[ 8  0  0  0  0  0  0  0  0  0]
 [ 0 14  0  0  0  0  0  0  0  0]
 [ 0  0  8  0  0  0  0  0  0  0]
 [ 0  0  0 10  0  0  0  0  1  0]
 [ 0  0  0  0 14  0  0  0  0  0]
 [ 0  0  0  0  0  7  0  0  0  0]
 [ 0  0  0  0  0  0 10  0  0  0]
 [ 0  0  0  0  0  0  0 15  0  0]
 [ 0  0  0  0  0  0  0  0  2  0]
 [ 0  0  0  0  0  0  0  0  0 11]]
[[0 2 1 1 0 1 1 0 2 0]
 [0 1 3 3 0 4 1 1 1 0]
 [1 2 1 1 0 0 1 0 1 1]
 [2 1 0 0 3 1 2 1 0 1]
 [0 1 3 0 4 1 2 3 0 0]
 [1 0 0 0 0 0 2 1 1 2]
 [2 0 1 2 1 1 1 1 0 1]
 [2 1 1 1 0 1 1 2 3 3]
 [0 0 0 1 0 0 0 0 1 0]
 [2 1 0 2 1 1 0 1 2 1]]
"""

