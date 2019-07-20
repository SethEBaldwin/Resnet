import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.backend.common import set_floatx

#set_floatx('float16')

L2_DECAY = .00005

def ShortcutConnection(filters, kernel_size, downsample, input):
	if downsample:
		conv0 = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, 
			strides = 2, padding = 'same', activation = 'relu', 
			kernel_regularizer = keras.regularizers.l2(L2_DECAY))
		batchnorm0 = keras.layers.BatchNormalization()
		activation0 = batchnorm0(conv0(input))
		conv1 = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, 
			strides = 1,  padding = 'same', 
			kernel_regularizer = keras.regularizers.l2(L2_DECAY))
		batchnorm1 = keras.layers.BatchNormalization()
		proj = keras.layers.Conv2D(filters = filters, kernel_size = 1, 
			strides = 2, padding = 'same')
		activation1 = batchnorm1(keras.layers.Activation('relu')(
			keras.layers.Add()([conv1(activation0), proj(input)])))
	else:
		conv0 = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, 
			strides = 1, padding = 'same', activation = 'relu', 
			kernel_regularizer = keras.regularizers.l2(L2_DECAY))
		batchnorm0 = keras.layers.BatchNormalization()
		conv1 = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, 
			strides = 1,  padding = 'same', 
			kernel_regularizer = keras.regularizers.l2(L2_DECAY))
		batchnorm1 = keras.layers.BatchNormalization()
		activation0 = batchnorm0(conv0(input))
		activation1 = batchnorm1(keras.layers.Activation('relu')(
			keras.layers.Add()([conv1(activation0), input])))
	return activation1
		
def ResNet(input_shape, output_size, shortcuts, downsample_idxs):
	input = keras.layers.Input(shape = input_shape)
	conv0 = keras.layers.Conv2D(filters = 16, kernel_size = 3, strides = 1, padding = 'same',
		kernel_regularizer = keras.regularizers.l2(L2_DECAY)) #64, 7, 2
	batchnorm = keras.layers.BatchNormalization()
	a = batchnorm(conv0(input))
	k = 0
	for i in range(shortcuts):
		if i in downsample_idxs:
			downsample = True
			k += 1
		else:
			downsample = False
		a = ShortcutConnection(filters = 16 * 2 ** k, kernel_size = 3, downsample = downsample, input = a)
	avgpool = keras.layers.GlobalAveragePooling2D()
	a = avgpool(a)
	softmax = keras.layers.Dense(output_size, activation = 'softmax')
	output = softmax(a)
	model = keras.models.Model(inputs = input, outputs = output)
	optimizer = keras.optimizers.SGD(lr = .1, momentum = .9)
	model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
	return model
		
n = 5
resnet = ResNet(input_shape = (32, 32, 3), output_size = 10, shortcuts = 3 * n, downsample_idxs = [n, 2 * n])
resnet.summary()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train.astype(np.int16)
x_test.astype(np.int16)

for (i, j, k) in zip(range(32), range(32), range(3)):
	x_train[:, i, j, k] -= np.mean(x_train[:, i, j, k]).astype(np.int16)
	x_test[:, i, j, k] -= np.mean(x_test[:, i, j, k]).astype(np.int16)

datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range = 8, 
	height_shift_range = 8, horizontal_flip = True, validation_split = .1) # rotation_range = 15, 
datagen.fit(x_train)

def schedule(epoch, lr):
	if epoch == 90: return lr / 10
	elif epoch == 135: return lr / 10
	else: return lr

reduce_lr = keras.callbacks.LearningRateScheduler(schedule, verbose = 0)
history = resnet.fit(datagen.flow(x_train, y_train, batch_size = 128, subset = 'training'), batch_size = 128, 
	epochs = 180, validation_data = datagen.flow(x_train, y_train, batch_size = 128, subset = 'validation'), 
	callbacks = [reduce_lr])

test_loss, test_acc = resnet.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

resnet.save('resnet_cifar_{}'.format(n))
print('Model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc = 'upper left')
plt.show()
