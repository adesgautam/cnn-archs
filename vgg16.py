import keras
import tensorflow as tf

def VGG16(classes=classes):
	model = keras.models.Sequential()

	model.add(keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(224,224,3)))
	model.add(keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

	model.add(keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

	model.add(keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

	model.add(keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

	model.add(keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(4096, activation='relu'))
	model.add(keras.layers.Dense(4096, activation='relu'))
	model.add(keras.layers.Dense(classes, activation='softmax'))

	opt = keras.optimizers.Adam(lr=0.001)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

	return model