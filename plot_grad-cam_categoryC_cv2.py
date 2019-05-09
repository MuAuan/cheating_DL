from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
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
from keras.models import Model
import matplotlib.pyplot as plt

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
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
    
        new_model = VGG16(weights='imagenet')
    
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

def grad_cam(input_model, image, category_index, layer_name):
    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    x = Lambda(target_layer, output_shape = target_category_loss_output_shape)(input_model.output)
    model = Model(inputs=input_model.input, outputs=x)
            
    loss = K.sum(model.layers[-1].output)
    conv_output =  [l for l in model.layers if l.name == layer_name][0].output  #is
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
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

model = VGG16(weights='imagenet')
model.summary()
        
def preprocessed(guided_model,preprocessed_input,s):
    predictions = model.predict(preprocessed_input)
    top_1 = decode_predictions(predictions)[0][s]
    print('Predicted class:')
    print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
    predicted_class = predictions.argsort()[0][::-1][s]

    cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
    saliency_fn = compile_saliency_function(guided_model)
    saliency = saliency_fn([preprocessed_input, 0])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    return predictions, top_1, cam, gradcam

s1=0
size=(224,224)
video_input = cv2.VideoCapture(0)

register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
#guided_model.summary()
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,4)
#frame = cv2.imread("cat_dog.png") #""dog_cat.png")
s = 0
while (1):
    print(s)
    ret, frame = video_input.read()
    input= cv2.resize(frame, (480,480))
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    cv2.imshow("original_"+str(s),input)

    preprocessed_input= cv2.resize(frame, size)
    preprocessed_input= np.expand_dims(preprocessed_input, axis=0)
    predictions, top_1, cam, gradcam = preprocessed(guided_model,preprocessed_input,s)

    input= cv2.resize(cam, (480,480))
    cv2.imshow("gradcam_"+str(top_1[1])+"_"+ str(int(top_1[2]*1000)/10)+" %",input)

    input= cv2.resize(deprocess_image(gradcam), (480,480))
    cv2.imshow("guided_gradcam_"+str(top_1[1])+"_"+ str(int(top_1[2]*1000)/10)+" %",input)
    k = cv2.waitKey(-1)&0xff
    if k == ord('0'):
        cv2.destroyAllWindows()
        s=0
    if k == ord('u'):
        cv2.destroyAllWindows()
        s=1
    elif k == ord('h'):
        cv2.destroyAllWindows()
        s=2
    elif k == ord('j'):
        cv2.destroyAllWindows()
        s=3
    elif k == ord('n'):
        cv2.destroyAllWindows()
        s=4
    elif k == ord('q'):
        cv2.destroyAllWindows()
        break  
        