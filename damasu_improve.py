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
        plt.title(titles[i], size=10)
        plt.xticks(())
        plt.yticks(())

# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = y_pred[i]  #target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[i]   #.rsplit(' ', 1)[-1]
    print(pred_name, true_name)
    return 'predicted: {}\ntrue:  {}'.format(pred_name, true_name)
        

# plot the result of the prediction on a portion of the test set
def title1(y_pred, y_test, target_names, i):
    pred_name = y_pred[i]  #target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[i]   #.rsplit(' ', 1)[-1]
    print(pred_name, true_name)
    return 'predicted: {}'.format(pred_name)



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


#def model_mnist(input_image=Input(shape=(None, None, 1))):
model = Sequential()
model.add(InputLayer(input_shape=(28,28,1)))
model.add(GaussianNoise(0.8))
model.add(Conv2D(32, (2, 2), activation="relu", padding="same"))  #,input_shape=(28,28,1)))
model.add(Conv2D(128, (2, 2), activation="relu", padding="same"))  #"valid"))
model.add(Conv2D(128, (1, 1), activation="relu", padding="same"))  #"valid"))
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
checkpointer = ModelCheckpoint(filepath='./cifar100/mnist_cnn_G08.hdf5', 
                               monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max',
                               verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                               factor=0.5, min_lr=0.00001, verbose=1)
csv_logger = CSVLogger('./cifar100/history_mnist_G08.log', separator=',', append=True)
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

#Learning ; Original x_train, y_train
history = model.fit(x_train, y_train,
          batch_size=64,
          epochs=10,
          callbacks=callbacks,          
          validation_split=0.2,
          shuffle=True) 


history = model.fit(x_train, y_train, epochs=2,
                    batch_size=64,
                    callbacks=callbacks,
                    validation_split=0.2)

model.save_weights('./cifar100/mnist_cnn_G08.hdf5', True)
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

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(x_test[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_test_G08.jpg')
plt.pause(1)
plt.close()

#check x_train[:10000] += Noise
predict_classes = model.predict_classes(x_train[:10000,], batch_size=32)
true_classes = np.argmax(y_train[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_train, true_classes, i) for i in range(81)]
plot_gallery(x_train[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_train_G08.jpg')
plt.pause(1)
plt.close()


#check x_adv[:10000] += Noise
predict_classes = model.predict_classes(x_adv[:10000,], batch_size=32)
true_classes = np.argmax(y_test[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(x_adv[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_adv_G08.jpg')
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
plt.pause(1)
plt.savefig('./cifar100/mnist_x_test_G08_data.jpg')
plt.close()

fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_train[:10000] += Noise
for i in range(W*H):
    ax1 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax1.imshow(x_train[i].reshape((28, 28)), cmap='gray')
plt.pause(1)
plt.savefig('./cifar100/mnist_x_train_G08_data.jpg')
plt.close()

fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_adv = x_test + Noise
for i in range(W*H):
    ax2 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax2.imshow(x_adv[i].reshape((28, 28)), cmap='gray')
plt.pause(1)
plt.savefig('./cifar100/mnist_x_adv_G08_data.jpg')
plt.close()




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
 G05
 [[ 973    0    2    0    0    1    2    1    1    0]
 [   0 1132    1    1    0    0    1    0    0    0]
 [   3    8 1006    0    4    0    1    8    2    0]
 [   1    0    3  990    0    5    0    8    1    2]
 [   1    2    1    0  958    0    5    0    0   15]
 [   3    1    1    2    0  878    6    1    0    0]
 [   7    4    0    0    2    3  941    0    1    0]
 [   1    7   13    1    0    1    0  993    2   10]
 [   7    2   10    3    5    2    6    7  912   20]
 [   3    6    0    2    9    5    0    8    1  975]]
 [[637   1 122   5   4  69  99  15  20  29]
 [  0 964  41   7  20   9  14  54  12   6]
 [ 26 138 416 155  30   2  12  63 135  14]
 [ 10  16 106 432   2 255   0  32  80  99]
 [ 16  22  16   0 313   4  21  18  16 554]
 [ 20  18   5 189   9 335  94   5 111  77]
 [ 93  36  24   1  43 133 667   1  10   6]
 [ 10  32  53  28  19   7   0 464   2 455]
 [ 23  71 192  63  35  90  65  15 201 189]
 [ 19   7   4  17 208  40   2 284  11 386]]
 [[ 612    0  128    6    4   61  108   11   19   31]
 [   1 1006   31    5   17    8   18   31   18    0]
 [  31  161  443  181   27    2    9   68   97   13]
 [  11    2  111  381    1  322    3   38   66   75]
 [   9   23   14    0  282    2   31   16   18  587]
 [  26   15    3  232    3  341   63    9  142   58]
 [ 117   13   18    1   56  126  612    1   12    2]
 [   7   39   75   30   17    5    0  451    8  396]
 [  32   34  205   52   27  103   71   34  190  226]
 [  13   11    5   23  211   63    0  261   22  400]]
 G05
 [[ 971    0    4    0    0    1    3    1    0    0]
 [   0 1131    1    1    0    0    2    0    0    0]
 [   5   15  990    6    1    0    3    6    6    0]
 [   0    0    1  996    0    4    0    5    1    3]
 [   1    2    1    0  962    0    5    0    2    9]
 [   2    1    0    5    0  876    7    0    1    0]
 [   5    4    0    0    1    1  947    0    0    0]
 [   1   12   14    2    0    0    0  986    2   11]
 [  13    2    3    5    4    6   10    6  908   17]
 [   7    6    0    3    8    3    0    6    1  975]]
 [[642   1  77  39   1  73 128   6  20  14]
 [  1 998  36  11  17   4  17  28  11   4]
 [ 38 131 328 275  31   1  15  55  98  19]
 [ 22  18  71 529   1 210   0  28  47 106]
 [ 13  25  28   3 353   0  31  23  11 493]
 [  9  21   1 277  10 271 121   2  62  89]
 [ 82  34   7   4  32  67 770   0  12   6]
 [  7  40  49  47  30   5   0 445   2 445]
 [ 47  87 109 115  28 121  85  10 192 150]
 [ 18  15   7  36 303  33   4 170  25 367]]
 [[ 634    0   80   17    2   64  148    6   12   17]
 [   1 1032   26    6    6    2   22   18   21    1]
 [  41  148  336  314   25    1   17   66   73   11]
 [  12   12   55  505    1  250    5   31   42   97]
 [   8   24   24    1  313    1   47   16    9  539]
 [  16   15    2  310    3  276  102    6   74   88]
 [ 111   16    8    2   47   59  704    1    8    2]
 [   3   64   57   47   21    3    0  420   10  403]
 [  60   45  112  129   28  125   96   21  167  191]
 [  14   11    4   40  315   45    6  165   23  386]]
 G01
 [[ 965    0    1    1    1    1    9    0    2    0]
 [   1 1125    1    0    2    0    6    0    0    0]
 [   3    2 1015    2    1    0    1    3    5    0]
 [   1    0    2  997    0    5    0    0    5    0]
 [   0    0    0    0  968    0    3    0    0   11]
 [   2    0    1    9    1  864    9    0    5    1]
 [   4    2    0    0    2    1  948    0    1    0]
 [   1    9   14    4    0    0    0  989    5    6]
 [   2    0    5    5    1    2    7    2  945    5]
 [   5    5    0    3    4    3    1    4    9  975]]
 [[  3   0 442  31   3  16 344   2 134  26]
 [  2  25 215  18 419   3  54  31 356   4]
 [ 40 119  24 367  45   1  11  65 301  18]
 [  6   8 204  33   2 346   6  41 265 121]
 [ 16  38  38  20  37   5  45  89 214 478]
 [ 17   2   1 239   4  31 165   0 281 123]
 [120  17  15  27 437 236 106   0  56   0]
 [ 11  83  79 361  27   6   0  19  40 444]
 [ 28  31 399 149  20 111  85  10   0 111]
 [ 17   3  13 165 189  27   2 280 282   0]]
 [[  0   0 408   9   3  16 368   0 154  22]
 [ 11  30 194  13 398   1  49  20 415   4]
 [ 34 131  33 429  35   1  12  81 262  14]
 [  5   6 232  28   3 375   7  36 209 109]
 [ 16  22  26  23  32   5  53  90 207 508]
 [ 17   0   2 267   1  21  97   1 348 138]
 [144   9  19  22 441 207  70   0  45   1]
 [  9  82  95 365  22   5   0  15  55 380]
 [ 32  12 408 130  13 136  88  33   3 119]
 [ 15   7   5 173 189  54   4 249 311   2]]
 
 G08
 [[ 971    0    1    0    0    0    5    1    2    0]
 [   0 1127    2    1    0    0    5    0    0    0]
 [  15   16  957    4    7    0    3   16   12    2]
 [   6    4    6  968    0    6    0   11    2    7]
 [   2   10    2    0  932    0    6    2    1   27]
 [   8    6    1    7    2  842   15    1    3    7]
 [   8    5    1    0    4    1  939    0    0    0]
 [   1   20   17    0    0    2    0  953    1   34]
 [  19    9    8    9    9    7   14   10  857   32]
 [  11   10    2    5    8    1    0   12    4  956]]
 [[695   4  32  36   2  70 117  15  15  15]
 [  0 988  32   8   6   7  19  38  14  15]
 [ 67 143 366 105  58   0  36  68 117  31]
 [ 47  43 105 422   7 192   9  32  60 115]
 [ 17  50  13   1 180   1  47  23   3 645]
 [ 49  44   6 246  36 107 128  19  97 131]
 [136  86  40   2  45  58 610   8  15  14]
 [ 16  49  24  16  36   6   1 380   1 541]
 [ 33 146 100 166  29  67  78  13 137 175]
 [ 22  30   4  22 225   5   4 269  10 387]]
 [[ 657    1   29   45    5   62  137   18   10   16]
 [   0 1009   35    9    1    4   18   22   32    5]
 [  71  152  387  137   47    0   36   76   87   39]
 [  39   35  101  410    7  220   16   35   61   86]
 [  10   35   12    0  124    0   35   19    7  740]
 [  52   34    3  252   28  152  109   26  115  121]
 [ 159   42   40    0   74   42  569    6   18    8]
 [  16   66   51   11   18    6    2  348    8  502]
 [  52   81  109  159   30   79   90   27  121  226]
 [  22   25    9   24  261   17    2  238   10  401]]
 


model.load_weights('./cifar100/mnist_cnn_G08.hdf5')

#es = keras.callbacks.EarlyStopping(monitor="val_acc", patience=1)
checkpointer = ModelCheckpoint(filepath='./cifar100/mnist_cnn_adv_G08.hdf5', 
                               monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max',
                               verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                               factor=0.5, min_lr=0.00001, verbose=1)
csv_logger = CSVLogger('./cifar100/history_mnist_adv_G08.log', separator=',', append=True)
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

history = model.fit(x_train, y_train,
          batch_size=64,
          epochs=20,
          callbacks=callbacks,          
          validation_split=0.2,
          shuffle=True) 

model.save_weights('./cifar100/mnist_cnn_adv_G08.hdf5', True)

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
plt.pause(1)
plt.savefig('./cifar100/mnist_x_test_input_G08_data.jpg')
plt.close()
fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_train[:50000] += Noise(1st) 
for i in range(W*H):
    ax1 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax1.imshow(x_train[i].reshape((28, 28)), cmap='gray')
plt.pause(1)
plt.savefig('./cifar100/mnist_x_train_input_G08_data.jpg')
plt.close()

fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_adv = x_test + Noise(1st)
for i in range(W*H):
    ax2 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax2.imshow(x_adv[i].reshape((28, 28)), cmap='gray')
plt.pause(1)
plt.savefig('./cifar100/mnist_x_adv_input_G08_data.jpg')
plt.close()

fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
#show x_adv_adv = x_adv+ Noise; x_test is original
for i in range(W*H):
    ax3 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax3.imshow(x_adv_adv[i].reshape((28, 28)), cmap='gray')
plt.pause(1)
plt.savefig('./cifar100/mnist_x_advadv_input_G08_data.jpg')
plt.close()


import numpy as np
from sklearn.metrics import confusion_matrix
#check x_test
#predict_classes = model.predict_classes(x_test[1:10000,], batch_size=32)
predict_classes = model.predict_classes(x_test[:10000,], batch_size=32)
true_classes = np.argmax(y_test[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(x_test[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_test_after_Tr_G08.jpg')
plt.pause(1)
plt.close()

#check x_train[:50000] += Noise(1st)
predict_classes = model.predict_classes(x_train[:10000,], batch_size=32)
true_classes = np.argmax(y_train[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(x_train[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_train_after_Tr_G08.jpg')
plt.pause(1)
plt.close()

#check x_adv = x_test + Noise(1st)
predict_classes = model.predict_classes(x_adv[:10000,], batch_size=32)
true_classes = np.argmax(y_test[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(x_adv[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_adv_after_Tr_G08.jpg')
plt.pause(1)
plt.close()

#check x_adv_adv = x_adv + Noise(2nd)
predict_classes = model.predict_classes(x_adv_adv[:10000,], batch_size=32)
true_classes = np.argmax(y_test[:10000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(x_adv_adv[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_x_adv_adv_after_Tr_G08.jpg')
plt.pause(1)
plt.close()


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
 
  G05
 [[962   0   1   0   0   0   9   0   7   1]
 [  0 955  30   1  69   0   2   4  73   1]
 [ 15   4 904   7  11   0  10  22  48  11]
 [  5   0  22 902   3   5   1   6  47  19]
 [  1   0   6   0 770   0  11   0   8 186]
 [ 13   1   1  74   5 508  37   0 214  39]
 [ 18   2   0   0   5   1 923   0   9   0]
 [  2   5  27  10   2   1   0 671   8 302]
 [  5   0   6   7   5   0   6   5 905  35]
 [  8   2   2   3   7   0   2   3  10 972]]
 [[1000    0    0    0    0    0    0    0    0    1]
 [   0 1127    0    0    0    0    0    0    0    0]
 [   0    0  987    0    1    0    1    1    0    1]
 [   0    0    1 1026    0    0    1    2    0    2]
 [   0    0    0    0  980    0    0    0    0    0]
 [   0    1    0    0    0  861    1    0    0    0]
 [   0    0    0    0    0    0 1014    0    0    0]
 [   0    1    0    1    1    0    0 1065    0    2]
 [   0    1    1    1    1    1    2    0  936    1]
 [   1    0    0    0    0    1    0    0    0  976]]
 [[ 977    0    0    0    0    1    0    0    1    1]
 [   0 1134    1    0    0    0    0    0    0    0]
 [   1    0 1025    0    0    0    0    1    1    4]
 [   0    0    0 1009    0    0    0    0    0    1]
 [   0    0    1    0  981    0    0    0    0    0]
 [   0    0    0    0    0  892    0    0    0    0]
 [   0    0    0    0    0    0  957    0    0    1]
 [   0    0    0    0    0    0    0 1028    0    0]
 [   1    0    0    2    0    0    2    0  969    0]
 [   0    0    1    2    2    1    2    1    0 1000]]
 [[682   2  20  44  29  27  29   3  96  48]
 [ 31 508  64  95 100  18  52   6 124 137]
 [ 45  10 550  91  57  10  40   3 151  75]
 [ 28   7  35 619  44  22  30   5 130  90]
 [ 24   4  20  60 571  15  30  16 125 117]
 [ 44   3  33  97  67 329  48   6 139 126]
 [ 26   5  29  58  73  11 606   3  86  61]
 [ 41   9  25 133 115  20  20 408 144 113]
 [ 35   7  39 109  63  24  53   8 550  86]
 [ 31   9  30  79  75  22  31   8 158 566]]
 
 G01
 [[ 952    0    0    0    1    0   10    0   16    1]
 [   0 1067    1    1    0    0    5    1   60    0]
 [   9   16  746   17    6    0   13    2  221    2]
 [   5    1    3  816    0    0    2    5  170    8]
 [   1    0    0    0  956    0    5    0   17    3]
 [  13    3    0   54    0  555   25    0  237    5]
 [   7    5    0    0    1    1  924    0   20    0]
 [   9    4    7    1    9    1    0  895   65   37]
 [   7    0    0    4    1    0    5    1  956    0]
 [  10    3    0    5   40    1    1    8   86  855]]
 [[1001    0    0    0    0    0    0    0    0    0]
 [   0 1127    0    0    0    0    0    0    0    0]
 [   0    0  991    0    0    0    0    0    0    0]
 [   0    0    0 1032    0    0    0    0    0    0]
 [   0    0    0    0  980    0    0    0    0    0]
 [   0    0    0    0    0  863    0    0    0    0]
 [   0    0    0    0    0    0 1014    0    0    0]
 [   0    1    0    0    0    0    0 1069    0    0]
 [   0    0    0    0    0    0    0    0  944    0]
 [   0    0    0    0    0    0    0    0    0  978]]
 [[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1135    0    0    0    0    0    0    0    0]
 [   0    1 1029    1    0    0    0    0    1    0]
 [   0    0    0 1010    0    0    0    0    0    0]
 [   0    0    0    0  982    0    0    0    0    0]
 [   1    0    1    1    0  887    0    0    1    1]
 [   0    0    0    1    0    0  957    0    0    0]
 [   1    2    1    1    0    0    0 1023    0    0]
 [   2    0    1    1    0    0    0    0  970    0]
 [   0    0    0    2    1    0    0    0    3 1003]]
 [[188  44  54  71  52  68  30  69 279 125]
 [125 123  55  89  58  68  40 100 288 189]
 [114  55 112  79  60  63  33  83 275 158]
 [119  53  54 111  52  76  36  81 266 162]
 [102  46  53  72 102  72  29  82 258 166]
 [111  39  58  71  51  91  29  71 225 146]
 [128  49  53  82  47  67  71  69 244 148]
 [129  63  43  87  52  58  23 135 277 161]
 [128  51  46  74  54  70  31  62 306 152]
 [119  51  52  66  54  92  31  78 270 196]]
 
 G08
 [[774   0  36   1   1 129  38   1   0   0]
 [  0 613 418   0  14  88   1   1   0   0]
 [  2   1 987   1   9  16   7   9   0   0]
 [  0   0 133 476   0 384   0  12   3   2]
 [  1   0  11   0 905  40   5   0   0  20]
 [  0   0  28  10  11 830  12   0   0   1]
 [  6   0  20   0  12 114 806   0   0   0]
 [  1   1 101   0  31  25   0 813   0  56]
 [  1   1 307   7  40 272  13  14 299  20]
 [  3   1  13   1 397 101   1  55   2 435]]
 [[ 998    1    0    0    1    0    0    0    0    1]
 [   0 1121    0    4    0    0    1    1    0    0]
 [   0    1  979    3    0    1    3    2    0    2]
 [   2    2    4 1008    2    0    1    6    0    7]
 [   1    0    0    0  978    0    0    0    1    0]
 [   1    0    0    1    0  860    1    0    0    0]
 [   0    0    0    0    0    1 1012    0    1    0]
 [   0    2    1    1    0    0    3 1057    1    5]
 [   2    0    3    2    3    1    5    6  918    4]
 [   2    0    0    1    0    2    2    2    1  968]]
 [[ 977    1    0    0    1    0    0    0    0    1]
 [   0 1132    1    1    0    1    0    0    0    0]
 [   3    0 1018    1    1    1    0    1    1    6]
 [   0    0    0 1000    0    1    0    1    0    8]
 [   1    1    0    0  979    0    1    0    0    0]
 [   1    0    1    0    0  888    1    0    0    1]
 [   0    0    0    0    0    2  955    0    0    1]
 [   0    3    3    1    0    0    1 1017    3    0]
 [   0    0    0    1    1    0    6    6  951    9]
 [   0    0    1    2    0    3    1    1    2  999]]
 [[749  49   5  18  29  49  45  13   2  21]
 [  0 937  16  27  14  55  16  16   4  50]
 [ 45 205 535  31  55  40  50  24   9  38]
 [ 20 157  21 604  49  38  51  18  10  42]
 [ 14  96   4  22 673  19  32  50   2  70]
 [ 30 111   5  46  53 493  51  26  10  67]
 [ 22  98  11  16  47  41 697   8   5  13]
 [ 20 113  13  25 116  18  33 648   5  37]
 [ 30 185  19  54  69  53  70  38 382  74]
 [ 16 101   6  11  71  11  37  19   3 734]]
 
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
        model.add(InputLayer(input_shape=(28,28,1)))
        model.add(GaussianNoise(0.8))
        model.add(keras.layers.Conv2D(32, (2, 2),
                                      activation="relu",
                                      padding="same"))  #,
                                      #input_shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(128, (2, 2),
                                      activation="relu",
                                      padding="same"))
        model.add(keras.layers.Conv2D(128, (1, 1),
                                      activation="relu",
                                      padding="same"))
        model.add(keras.layers.Flatten())
        # softmaxは適用しない
        model.add(keras.layers.Dense(10))

        # 訓練済みモデルのロード
        model.load_weights('./cifar100/mnist_cnn_G08.hdf5')

        self.model = model

    def predict(self, data):
        return self.model(data)


N = 1000

model = Sequential()
model.add(InputLayer(input_shape=(28,28,1)))
model.add(GaussianNoise(0.8))
model.add(Conv2D(32, (2, 2), activation="relu", padding="same")) 
model.add(Conv2D(128, (2, 2), activation="relu", padding="same"))
model.add(Conv2D(128, (1, 1), activation="relu", padding="same"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))
model.summary()

#model = model_mnist(input_image=Input(shape=(28, 28, 1)))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["acc"])
model.load_weights('./cifar100/mnist_cnn_G08.hdf5')

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
plt.savefig('./cifar100/mnist_x_test100_G08.jpg')
plt.pause(1)
plt.close()

fig = plt.figure(figsize=(H, W))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
for i in range(W*H):
    ax2 = fig.add_subplot(H, W, i + 1, xticks=[], yticks=[])
    ax2.imshow(adv[i].reshape((28, 28)), cmap='gray')
plt.savefig('./cifar100/mnist_adv100_G08.jpg')
plt.pause(1)
plt.close()

import numpy as np
from sklearn.metrics import confusion_matrix

predict_classes = model.predict_classes(x_test[:1000,], batch_size=32)
true_classes = np.argmax(y_test[:1000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title1(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(x_test[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_L2_attack_x_test81_G08.jpg')
plt.pause(1)
plt.close()

predict_classes = model.predict_classes(adv[:1000,], batch_size=32)
true_classes = np.argmax(y_test[:1000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title1(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(adv[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_L2_attack_adv81_G08.jpg')
plt.pause(1)
plt.close()

x_train=adv[:1000]
y_train=y_test[:1000]

model.load_weights('./cifar100/mnist_cnn_G08.hdf5')

#es = keras.callbacks.EarlyStopping(monitor="val_acc", patience=1)
checkpointer = ModelCheckpoint(filepath='./cifar100/mnist_cnn_advadv_G08.hdf5', 
                               monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max',
                               verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=5,
                               factor=0.5, min_lr=0.00001, verbose=1)
csv_logger = CSVLogger('./cifar100/history_mnist_advadv_G08.log', separator=',', append=True)
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer]

history = model.fit(x_train, y_train,
          batch_size=64,
          epochs=20,
          callbacks=callbacks,          
          validation_split=0.2,
          shuffle=True) 

model.save_weights('./cifar100/mnist_cnn_advadv_G08.hdf5', True)

class MNIST_data:
    def __init__(self):
        self.test_data = adv[:1000]
        self.test_labels = y_test[:1000]

with tf.Session() as sess:
    data, model1 = MNIST_data(), MNISTModel()
    attack = CarliniL2(sess, model1, batch_size=100, max_iterations=1000,
                       confidence=0, boxmin=0, boxmax=1)

    inputs, targets = generate_data(data, samples=N, targeted=True,
                                    start=0, inception=False)
    print(targets)
    adv_adv = attack.attack(inputs, targets)

predict_classes = model.predict_classes(adv_adv[:1000,], batch_size=32)
true_classes = np.argmax(y_test[:1000],1)
print(confusion_matrix(true_classes, predict_classes))

prediction_titles = [title1(predict_classes, y_test, true_classes, i) for i in range(81)]
plot_gallery(adv_adv[:81], prediction_titles, 28, 28)
plt.savefig('./cifar100/mnist_L2_attack_advadv81_G08.jpg')
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
 G05
 [[ 8  0  0  0  0  0  0  0  0  0]
 [ 0 11  0  0  0  0  0  0  0  0]
 [ 0  0  7  0  0  0  0  0  0  0]
 [ 0  0  0  8  0  0  0  0  0  0]
 [ 0  0  0  0 12  0  0  0  0  0]
 [ 0  0  0  0  0  6  1  0  0  0]
 [ 0  0  0  0  0  0  6  0  0  0]
 [ 0  0  0  0  0  0  0 12  0  0]
 [ 0  0  0  0  0  0  0  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  9]]
 [[1 1 2 0 1 0 0 1 0 2]
 [1 2 2 2 1 1 0 2 0 0]
 [1 1 1 0 0 1 1 1 1 0]
 [2 2 0 0 2 0 0 0 1 1]
 [1 4 0 0 1 1 0 3 0 2]
 [0 4 1 0 1 0 0 1 0 0]
 [0 3 1 1 1 0 0 0 0 0]
 [1 2 1 2 2 0 0 2 1 1]
 [0 0 0 0 1 0 0 0 0 0]
 [0 2 2 0 3 0 0 1 1 0]]
 G01
 [[ 8  0  0  0  0  0  0  0  0  0]
 [ 0 11  0  0  0  0  0  0  0  0]
 [ 0  0  7  0  0  0  0  0  0  0]
 [ 0  0  0  7  0  0  0  0  1  0]
 [ 0  0  0  0 12  0  0  0  0  0]
 [ 0  0  0  0  0  6  1  0  0  0]
 [ 0  0  0  0  0  0  6  0  0  0]
 [ 0  0  0  0  0  0  0 12  0  0]
 [ 0  0  0  0  0  0  0  0  1  0]
 [ 0  0  0  0  0  0  0  0  0  9]]
 [[0 2 1 1 0 1 1 0 2 0]
 [0 1 3 3 0 3 1 0 0 0]
 [1 1 1 1 0 0 1 0 1 1]
 [2 0 0 0 2 1 1 1 0 1]
 [0 1 3 0 3 1 1 3 0 0]
 [1 0 0 0 0 0 2 1 1 2]
 [1 0 0 2 1 1 1 0 0 0]
 [2 1 0 1 0 0 1 2 2 3]
 [0 0 0 0 0 0 0 0 1 0]
 [1 1 0 1 1 1 0 1 2 1]]
 G08
 [[ 8  0  0  0  0  0  0  0  0  0]
 [ 0 11  0  0  0  0  0  0  0  0]
 [ 0  0  6  0  0  0  0  1  0  0]
 [ 0  0  1  7  0  0  0  0  0  0]
 [ 1  0  0  0 11  0  0  0  0  0]
 [ 0  0  0  0  0  6  1  0  0  0]
 [ 0  0  0  0  0  0  6  0  0  0]
 [ 0  0  0  0  0  0  0 11  0  1]
 [ 0  0  0  0  0  0  0  0  1  0]
 [ 0  0  0  0  0  0  0  1  0  8]]
 [[0 2 1 1 0 1 1 0 2 0]
 [0 1 3 3 0 3 1 0 0 0]
 [1 1 1 1 0 0 1 0 1 1]
 [2 0 0 0 2 1 1 1 0 1]
 [0 1 3 0 3 1 1 3 0 0]
 [1 0 0 0 0 0 2 1 1 2]
 [1 0 0 2 1 1 1 0 0 0]
 [2 1 0 1 0 0 1 2 2 3]
 [0 0 0 0 0 0 0 0 1 0]
 [1 1 0 1 1 1 0 1 2 1]]
 G08_advadv
 [[1 7 0 0 0 0 0 0 0 0]
 [1 8 0 0 0 1 0 1 0 0]
 [1 2 0 1 2 1 0 0 0 0]
 [0 5 0 0 2 1 0 0 0 0]
 [0 7 0 0 1 1 0 3 0 0]
 [1 4 0 0 0 2 0 0 0 0]
 [0 5 0 0 0 0 1 0 0 0]
 [1 8 0 0 0 1 0 2 0 0]
 [0 1 0 0 0 0 0 0 0 0]
 [0 8 0 0 0 0 0 1 0 0]]
"""

