import tensorflow as tf
import keras

from keras.layers import *

def fire_block(i, x, squeeze_f, expand_f):
    squeezed = Conv2D(squeeze_f, kernel_size=(1,1), padding='same', strides=(1,1), activation='relu', name='fire_squeeze_conv_'+i)(x)
    expanded_1x1 = Conv2D(expand_f, kernel_size=(1,1), padding='same', strides=(1,1), activation='relu', name='fire_expand_conv_'+i+'a')(squeezed)    
    expanded_3x3 = Conv2D(expand_f, kernel_size=(3,3), padding='same', strides=(1,1), activation='relu', name='fire_expand_conv_'+i+'b')(squeezed)        
    x = Concatenate(axis=3)([expanded_1x1, expanded_3x3])
    return x

def squeezenet(classes=1000):
	inp = Input(shape=(224,224,3), name='input_layer')
	x = Conv2D(96, kernel_size=(7,7), padding='same', strides=(2,2), activation='relu', name='conv_1')(inp)
	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool_1')(x)
	print(x.shape)

	x = fire_block('1', x, 16, 64)
	x = fire_block('2', x, 16, 64)
	x = fire_block('3', x, 32, 128)

	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool_2')(x)

	x = fire_block('4', x, 32, 128)
	x = fire_block('5', x, 48, 192)
	x = fire_block('6', x, 48, 192)
	x = fire_block('7', x, 64, 256)

	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool_3')(x)
	x = fire_block('8', x, 64, 256)
	x = Conv2D(1000, kernel_size=(13,13), padding='same', strides=(1,1), activation='relu', name='conv_2')(x)

	out = AveragePooling2D(pool_size=(13,13), name='avgpool_1')(x)
	out = Dense(classes, activation='softmax', name='dense_1')(out)

	model = keras.Model(inputs=inp, outputs=out, name="squeezenet_model")

	opt = keras.optimizers.Adam(lr=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model