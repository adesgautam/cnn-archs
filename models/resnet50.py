import tensorflow as tf
import keras

from keras.layers import *

def conv_block(base, filters, kernel_size, padding, strides, name):
    x = Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides, name='conv'+name)(base)
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name='batchnorm'+name)(x)
    x = Activation('relu', name='act'+name)(x)
    return x

def resnet50(classes=1000):
	inp = Input(shape=(224,224,3), name='input_layer')
	x = ZeroPadding2D(padding=(3, 3), name='conv0_pad')(inp)
	x = Conv2D(64, kernel_size=(7,7), padding='valid', strides=(2,2), activation='relu', name='conv_0')(x)
	x = ZeroPadding2D(padding=(1, 1), name='maxpool0_pad')(x)
	base = MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool_0')(x)

	print('Stage 0:', base.shape)

	# Stage 1 (3 cnv_blocks)
	names = ['_1_a', '_1_b', '_1_c']
	for n in names:
	    x = conv_block(base, 64, kernel_size=(1,1), padding='same', strides=(1,1), name=n+'1')
	    x = conv_block(x, 64, kernel_size=(3,3), padding='same', strides=(1,1), name=n+'2')
	    x1 = conv_block(x, 256, kernel_size=(1,1), padding='same', strides=(1,1), name=n+'3')

	    # Shortcut
	    x = Conv2D(256, kernel_size=(1,1), padding='same', strides=(1,1), name='conv'+n+'4')(base)
	    shortcut = BatchNormalization(axis=3, epsilon=1.001e-5, name='batchnorm'+n+'4')(x)
	    base = Add(name='add'+n+'1')([x1, shortcut])
	    base = Activation('relu', name='act'+n+'4')(base)

	print('Stage 1:', base.shape)

	# Stage 2 (4 cnv_blocks)
	names = ['_2_a', '_2_b', '_2_c', '_2_d']
	for n in names:
	    if n=='_2_a':
	        x = conv_block(base, 128, kernel_size=(1,1), padding='same', strides=(2,2), name=n+'1')
	        conv_shortcut = Conv2D(512, kernel_size=(1,1), padding='same', strides=(2,2), name='conv'+n+'4')(base)
	    else:
	        x = conv_block(base, 128, kernel_size=(1,1), padding='same', strides=(1,1), name=n+'1')
	        conv_shortcut = Conv2D(512, kernel_size=(1,1), padding='same', strides=(1,1), name='conv'+n+'4')(base)
	        
	    x = conv_block(x, 128, kernel_size=(3,3), padding='same', strides=(1,1), name=n+'2')
	    x1 = conv_block(x, 512, kernel_size=(1,1), padding='same', strides=(1,1), name=n+'3')

	    # Shortcut
	    shortcut = BatchNormalization(axis=3, epsilon=1.001e-5, name='batchnorm'+n+'4')(conv_shortcut)
	    base = Add(name='add'+n+'1')([x1, shortcut])
	    base = Activation('relu', name='act'+n+'4')(base)

	print('Stage 2:', base.shape)

	# Stage (6 cnv_blocks)
	names = ['_3_a', '_3_b', '_3_c', '_3_d', '_3_e', '_3_f']
	for n in names:
	    if n=='_3_a':
	        x = conv_block(base, 256, kernel_size=(1,1), padding='same', strides=(2,2), name=n+'1')
	        conv_shortcut = Conv2D(1024, kernel_size=(1,1), padding='same', strides=(2,2), name='conv'+n+'4')(base)
	    else:
	        x = conv_block(base, 256, kernel_size=(1,1), padding='same', strides=(1,1), name=n+'1')
	        conv_shortcut = Conv2D(1024, kernel_size=(1,1), padding='same', strides=(1,1), name='conv'+n+'4')(base)
	        
	    x = conv_block(x, 256, kernel_size=(3,3), padding='same', strides=(1,1), name=n+'2')
	    x1 = conv_block(x, 1024, kernel_size=(1,1), padding='same', strides=(1,1), name=n+'3')

	    # Shortcut
	    shortcut = BatchNormalization(axis=3, epsilon=1.001e-5, name='batchnorm'+n+'4')(conv_shortcut)
	    base = Add(name='add'+n+'1')([x1, shortcut])
	    base = Activation('relu', name='act'+n+'4')(base)

	print('Stage 3:', base.shape)

	# Stage (3 cnv_blocks)
	names = ['_4_a', '_4_b', '_4_c']
	for n in names:
	    if n=='_4_a':
	        x = conv_block(base, 512, kernel_size=(1,1), padding='same', strides=(2,2), name=n+'1')
	        conv_shortcut = Conv2D(2048, kernel_size=(1,1), padding='same', strides=(2,2), name='conv'+n+'4')(base)
	    else:
	        x = conv_block(base, 512, kernel_size=(1,1), padding='same', strides=(1,1), name=n+'1')
	        conv_shortcut = Conv2D(2048, kernel_size=(1,1), padding='same', strides=(1,1), name='conv'+n+'4')(base)
	        
	    x = conv_block(x, 512, kernel_size=(3,3), padding='same', strides=(1,1), name=n+'2')
	    x1 = conv_block(x, 2048, kernel_size=(1,1), padding='same', strides=(1,1), name=n+'3')

	    # Shortcut
	    shortcut = BatchNormalization(axis=3, epsilon=1.001e-5, name='batchnorm'+n+'4')(conv_shortcut)
	    base = Add(name='add'+n+'1')([x1, shortcut])
	    base = Activation('relu', name='act'+n+'4')(base)

	print('Stage 4:', base.shape)

	out = GlobalAveragePooling2D(name='global_avg_1')(base)
	out = Dense(1000, activation='softmax', name='dense_1')(out)

	model = keras.Model(inputs=inp, outputs=out, name="resnet50_model")

	opt = keras.optimizers.Adam(lr=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model