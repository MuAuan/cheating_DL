from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.models import Model
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
from keras.datasets import cifar10,cifar100


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(64, 64))  #(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = model #VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 10
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
    model.summary()
    loss = K.sum(model.output)
    conv_output =  [l for l in model.layers if l.name is layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (64, 64))  #(224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.callbacks import ModelCheckpoint

from keras.layers import *
from keras.models import Model, Sequential
from keras.layers import Input, Dense

(x_train, y_train), (x_test, y_test) = cifar10.load_data() #mnist.load_data() #cifar10.load_data()
#x_train,y_train,x_test,y_test = getDataSet(img_rows,img_cols)
num_classes=10
img_rows,img_cols,ch=64,64,3

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

X_train = X_train.reshape(50000,img_rows, img_cols,ch)
X_test = X_test.reshape(10000,img_rows, img_cols,ch)   

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


preprocessed_input = load_image(sys.argv[1])
#print(preprocessed_input)
#model = VGG16(weights='imagenet')


preprocessed_input=preprocessed_input.reshape(1,img_rows,img_cols,3)

input_tensor = Input(shape=(img_rows,img_cols,3)) 
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC層を構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:])) 
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.75))
top_model.add(Dense(num_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=top_model(vgg16.output))
#model.load_weights('./mnist/cifar10_cnn.hdf5')

# 最後のconv層の直前までの層をfreeze
#trainingするlayerを指定　VGG16では18,15,10,1など 20で全層固定
for layer in model.layers[1:1]:  
    layer.trainable = False


# Fine-tuningのときはSGDの方がよい⇒adamがよかった
lr = 0.00001
opt = keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# モデルのサマリを表示
model.summary()
#model.load_weights('./cifar10/cifar10_cnn64.hdf5')

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test))

model.save_weights('./cifar10/weights_cifar10_initial_64.hdf5', True) 

checkpointer = ModelCheckpoint(filepath='./mnist/cifar10_cnn64.hdf5', 
                               monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_acc', patience=25, mode='max',
                               verbose=1)
lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=25,
                               factor=0.5, min_lr=0.00001, verbose=1)
csv_logger = CSVLogger('./mnist/history_cifar10_cnn64.log', separator=',', append=True)
#ch_layer = Check_layer()
callbacks = [early_stopping, lr_reduction, csv_logger,checkpointer] #,ch_layer]

#Learning ; Original x_train, y_train
history = model.fit(x_train, y_train,
          batch_size=32,
          epochs=100,
          callbacks=callbacks,          
          validation_split=0.2,
          shuffle=True) 

predictions = model.predict(preprocessed_input)
#top_1 = decode_predictions(predictions)[0][0]
print('Predicted class:')
#print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
print('probability {}'.format(np.argmax(predictions)))

predicted_class = np.argmax(predictions)
cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
cv2.imwrite("gradcam.jpg", cam)

register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
saliency_fn = compile_saliency_function(guided_model)
saliency = saliency_fn([preprocessed_input, 0])
gradcam = saliency[0] * heatmap[..., np.newaxis]
cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
