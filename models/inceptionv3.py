import tensorflow as tf
import keras

from keras.layers import *

def conv_block(base, filters, kernel_size, padding, strides, name):
    x = Conv2D(filters, kernel_size=kernel_size, padding=padding, strides=strides, name='conv'+name)(base)
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name='batchnorm'+name)(x)
    x = Activation('relu', name='act'+name)(x)
    return x

def inceptionv3(classes=1000):
	inp = Input(shape=(299,299,3), name='input_layer')

	x = conv_block(inp, filters=32, kernel_size=(3,3), padding='valid', strides=(2,2), name='conv_1')
	x = conv_block(x, filters=32, kernel_size=(3,3), padding='valid', strides=(1,1), name='conv_2')
	x = conv_block(x, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='conv_3')
	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool_1')(x)
	x = conv_block(x, filters=80, kernel_size=(3,3), padding='same', strides=(1,1), name='conv_4')
	x = conv_block(x, filters=192, kernel_size=(3,3), padding='valid', strides=(1,1), name='conv_5')
	base = MaxPool2D(pool_size=(3,3), strides=(2,2), name='maxpool_2')(x)

	# Inception block 1
	# Branch 1
	block1_br1_c1 = conv_block(base, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block1_br1_c1')
	block1_br1_c2 = conv_block(block1_br1_c1, filters=96, kernel_size=(3,3), padding='same', strides=(1,1), name='block1_br1_c2')
	branch1 = conv_block(block1_br1_c2, filters=96, kernel_size=(3,3), padding='same', strides=(1,1), name='block1_br1_c3')
	# Branch 2
	block1_br2_c1 = conv_block(base, filters=48, kernel_size=(3,3), padding='same', strides=(1,1), name='block1_br2_c1')
	branch2 = conv_block(block1_br2_c1, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block1_br2_c2')
	# Branch 3
	block1_br3_ap1 = AveragePooling2D(pool_size=(1,1), name='block1_br3_ap1')(base)
	branch3 = conv_block(block1_br3_ap1, filters=32, kernel_size=(3,3), padding='same', strides=(1,1), name='block1_br3_c1')
	# Branch 4
	branch4 = conv_block(base, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block1_br4_c1')
	# Concatenate
	block1 = Concatenate(axis=3)([branch1, branch2, branch3, branch4])
	print(block1.shape)

	# Inception block 2
	# Branch 1
	block2_br1_c1 = conv_block(block1, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block2_br1_c1')
	block2_br1_c2 = conv_block(block2_br1_c1, filters=96, kernel_size=(3,3), padding='same', strides=(1,1), name='block2_br1_c2')
	branch1 = conv_block(block2_br1_c2, filters=96, kernel_size=(3,3), padding='same', strides=(1,1), name='block2_br1_c3')
	# Branch 2
	block2_br2_c1 = conv_block(block1, filters=48, kernel_size=(3,3), padding='same', strides=(1,1), name='block2_br2_c1')
	branch2 = conv_block(block2_br2_c1, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block2_br2_c2')
	# Branch 3
	block2_br3_ap1 = AveragePooling2D(pool_size=(1,1), name='block2_br3_ap1')(block1)
	branch3 = conv_block(block1_br3_ap1, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block2_br3_c1')
	# Branch 4
	branch4 = conv_block(block1, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block2_br4_c1')
	# Concatenate
	block2 = Concatenate(axis=3)([branch1, branch2, branch3, branch4])
	print(block2.shape)

	# Inception block 3
	# Branch 1
	block3_br1_c1 = conv_block(block2, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block3_br1_c1')
	block3_br1_c2 = conv_block(block3_br1_c1, filters=96, kernel_size=(3,3), padding='same', strides=(1,1), name='block3_br1_c2')
	branch1 = conv_block(block3_br1_c2, filters=96, kernel_size=(3,3), padding='same', strides=(1,1), name='block3_br1_c3')
	# Branch 2
	block3_br2_c1 = conv_block(block2, filters=48, kernel_size=(3,3), padding='same', strides=(1,1), name='block3_br2_c1')
	branch2 = conv_block(block3_br2_c1, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block3_br2_c2')
	# Branch 3
	block3_br3_ap1 = AveragePooling2D(pool_size=(1,1), name='block3_br3_ap1')(block2)
	branch3 = conv_block(block3_br3_ap1, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block3_br3_c1')
	# Branch 4
	branch4 = conv_block(block2, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block3_br4_c1')
	# Concatenate
	block3 = Concatenate(axis=3)([branch1, branch2, branch3, branch4])
	print(block3.shape)

	# Inception Block 4
	# Branch 1
	block4_br1_c1 = conv_block(block3, filters=64, kernel_size=(3,3), padding='same', strides=(1,1), name='block4_br1_c1')
	block4_br1_c2 = conv_block(block4_br1_c1, filters=96, kernel_size=(3,3), padding='same', strides=(1,1), name='block4_br1_c2')
	branch1 = conv_block(block4_br1_c2, filters=96, kernel_size=(3,3), padding='valid', strides=(2,2), name='block4_br1_c3')
	# Branch 2
	branch2 = conv_block(block3, filters=384, kernel_size=(3,3), padding='valid', strides=(2,2), name='block4_br2_c3')
	# Branch 3
	branch3 = MaxPool2D(pool_size=(3,3), strides=(2,2), name='block4_br3_mp1')(block3)
	# Concatenate
	block4 = Concatenate(axis=3)([branch1, branch2, branch3])
	print(block4.shape)

	# Inception Block 5
	# Branch 1
	block5_br1_c1 = conv_block(block4, filters=128, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br1_c1')
	block5_br1_c2 = conv_block(block5_br1_c1, filters=128, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br1_c2')
	block5_br1_c3 = conv_block(block5_br1_c2, filters=128, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br1_c3')
	block5_br1_c4 = conv_block(block5_br1_c3, filters=128, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br1_c4')
	branch1 = conv_block(block5_br1_c4, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br1_c5')
	# Branch 2
	block5_br2_c1 = conv_block(block4, filters=128, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br2_c1')
	block5_br2_c2 = conv_block(block5_br2_c1, filters=128, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br2_c2')
	branch2 = conv_block(block5_br2_c2, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br2_c3')
	# Branch 3
	block5_br3_ap1 = AveragePooling2D(pool_size=(1,1), name='block5_br3_mp1')(block4)
	branch3 = conv_block(block5_br3_ap1, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br3_c1')
	# Branch 4
	branch4 = conv_block(block4, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block5_br4_c1')
	# Concatenate
	block5 = Concatenate(axis=3)([branch1, branch2, branch3, branch4])
	print(block5.shape)

	block6_7 = block5

	# Inception Block 6, 7
	for i in range(2):
	    # Branch 1
	    block6_br1_c1 = conv_block(block6_7, filters=160, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br1_c1'+str(i))
	    block6_br1_c2 = conv_block(block6_br1_c1, filters=160, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br1_c2'+str(i))
	    block6_br1_c3 = conv_block(block6_br1_c2, filters=160, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br1_c3'+str(i))
	    block6_br1_c4 = conv_block(block6_br1_c3, filters=160, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br1_c4'+str(i))
	    branch1 = conv_block(block6_br1_c4, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br1_c5'+str(i))
	    # Branch 2
	    block6_br2_c1 = conv_block(block6_7, filters=160, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br2_c1'+str(i))
	    block6_br2_c2 = conv_block(block6_br2_c1, filters=160, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br2_c2'+str(i))
	    branch2 = conv_block(block6_br2_c2, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br2_c3'+str(i))
	    # Branch 3
	    block6_br3_ap1 = AveragePooling2D(pool_size=(1,1), name='block6_7_br3_mp1'+str(i))(block6_7)
	    branch3 = conv_block(block6_br3_ap1, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br3_c1'+str(i))
	    # Branch 4
	    branch4 = conv_block(block6_7, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block6_7_br4_c1'+str(i))
	    # Concatenate
	    block6_7 = Concatenate(axis=3)([branch1, branch2, branch3, branch4])
	    print(block6_7.shape)
	    
	# Inception Block 8    
	# Branch 1
	block8_br1_c1 = conv_block(block6_7, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br1_c1')
	block8_br1_c2 = conv_block(block8_br1_c1, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br1_c2')
	block8_br1_c3 = conv_block(block8_br1_c2, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br1_c3')
	block8_br1_c4 = conv_block(block8_br1_c3, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br1_c4')
	branch1 = conv_block(block8_br1_c4, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br1_c5')
	# Branch 2
	block8_br2_c1 = conv_block(block6_7, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br2_c1')
	block8_br2_c2 = conv_block(block8_br2_c1, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br2_c2')
	branch2 = conv_block(block8_br2_c2, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br2_c3')
	# Branch 3
	block8_br3_ap1 = AveragePooling2D(pool_size=(1,1), name='block8_br3_mp1')(block6_7)
	branch3 = conv_block(block8_br3_ap1, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br3_c1')
	# Branch 4
	branch4 = conv_block(block6_7, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block8_br4_c1')
	# Concatenate
	block8 = Concatenate(axis=3)([branch1, branch2, branch3, branch4])
	print(block8.shape)

	# Inception Block 9  
	# Branch 1
	block9_br1_c1 = conv_block(block8, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block9_br1_c1')
	block9_br1_c2 = conv_block(block9_br1_c1, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block9_br1_c2')
	block9_br1_c3 = conv_block(block9_br1_c2, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block9_br1_c3')
	block9_br1_c4 = conv_block(block9_br1_c3, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block9_br1_c4')
	branch1 = conv_block(block9_br1_c4, filters=192, kernel_size=(3,3), padding='valid', strides=(2,2), name='block9_br1_c5')
	# Branch 2
	block9_br2_c1 = conv_block(block8, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block9_br2_c1')
	branch2 = conv_block(block9_br2_c1, filters=320, kernel_size=(3,3), padding='valid', strides=(2,2), name='block9_br2_c2')
	# Branch 3
	branch3 = MaxPool2D(pool_size=(2,2), name='block9_br3_mp1')(block8)
	# Concatenate
	block9 = Concatenate(axis=3)([branch1, branch2, branch3])
	print(block9.shape)

	block10_11 = block9
	    
	# Inception Block 10, 11
	for i in range(2):
	    # Branch 1
	    block10_br1_c1 = conv_block(block10_11, filters=448, kernel_size=(3,3), padding='same', strides=(1,1), name='block10_br1_c1'+str(i))
	    block10_br1_c2 = conv_block(block10_br1_c1, filters=384, kernel_size=(3,3), padding='same', strides=(1,1), name='block10_br1_c2'+str(i))
	    branch1 = conv_block(block10_br1_c2, filters=384, kernel_size=(3,3), padding='same', strides=(1,1), name='block10_br1_c3'+str(i))
	    # Branch 1a
	    branch1a = conv_block(block10_br1_c2, filters=384, kernel_size=(3,3), padding='same', strides=(1,1), name='block10_br1a_c1'+str(i))
	    # Branch 1 (concat with 1a)
	    branch1 = Concatenate(axis=3)([branch1, branch1a])
	    # Branch 2
	    block10_br2_c1 = conv_block(block10_11, filters=384, kernel_size=(3,3), padding='same', strides=(1,1), name='block10_br2_c1'+str(i))
	    # Branch 2a
	    branch2a = conv_block(block10_br2_c1, filters=384, kernel_size=(3,3), padding='same', strides=(1,1), name='block10_br2a_c1'+str(i))
	    # Branch 2b
	    branch2b = conv_block(block10_br2_c1, filters=384, kernel_size=(3,3), padding='same', strides=(1,1), name='block10_br2b_c1'+str(i))
	    # Branch 2(concat 2a, 2b)
	    branch2 = Concatenate(axis=3)([branch2a, branch2b])
	    # Branch 3
	    block10_br3_ap1 = AveragePooling2D(pool_size=(1,1), name='block10_br3_ap1'+str(i))(block10_11)
	    branch3 = conv_block(block10_br3_ap1, filters=192, kernel_size=(3,3), padding='same', strides=(1,1), name='block10_br3_c1'+str(i))
	    # Branch 4
	    branch4 = conv_block(block10_11, filters=320, kernel_size=(3,3), padding='same', strides=(1,1), name='block10_br4_c1'+str(i))
	    # Concatenate
	    block10_11 = Concatenate(axis=3)([branch1, branch2, branch3, branch4])
	    print(block10.shape)


	out = GlobalAveragePooling2D(name='global_avg_1')(block10_11)
	out = Dense(classes, activation='softmax', name='dense_1')(out)

	model = keras.Model(inputs=inp, outputs=out, name="inceptionv3_model")

	opt = keras.optimizers.Adam(lr=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



