from __future__ import print_function
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import keras

batch_size = 128
num_classes = 10
epochs = 10

# Constants for dimensions of our input images
X_DIM = 28
Y_DIM = 28

# mnist data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Here we reshape the data to a 4-dimensional vector
# (num_samples, x_dimension, y_dimesnion, number of "channels")
#
# NOTE: channels = 1 here because of grayscale, normally would be 3 for RBG
x_train = x_train.reshape(x_train.shape[0], X_DIM, Y_DIM, 1)
x_test = x_test.reshape(x_test.shape[0], X_DIM, Y_DIM, 1)

# Not sure what this is doing
input_shape = (X_DIM, Y_DIM, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Also not sure what this is doing
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CNN

model = Sequential() #declare Keras "Sequential" type of model, this is what most deep learning uses

# Adds 2D convolutional layer to our initialized model
# UNCLEAR WHY 32 is the output channel dimension??
# Kernel tells us the size of moving window, this can be tweaked
# Strides tells us x/y movement of window, this can also be tweaked
# Activation tells us type of activation function, this can be tweaked
# Input shape tells the model what kind of input to expected initially (need not include later)
model.add(Conv2D(32,
				kernel_size=(5,5),
				strides=(1,1),
				activation='relu',
				input_shape=input_shape))

# Adds pooling layer to our model w/ pool size and stride size of pool
model.add(MaxPooling2D(pool_size=(2, 2),
						strides=(2, 2)))

# Add another 2D layer to model, abbreviated syntax
model.add(Conv2D(64,
				(5, 5),
				activation='relu'))
# Add another pooling layer to model
# NOTE: Default strides is (1,1)
model.add(MaxPooling2D(pool_size=(2, 2)))

# "Flatten" input from previous layer into 1D vector for classifying
model.add(Flatten())
# Adds fully-connected layer (not sure how 1000 is calculated)
model.add(Dense(1000, activation='relu'))
# Finishes up model with fully connected layer and 'softmax' classification
model.add(Dense(num_classes, activation='softmax'))


# HAVE NOT GONE THROUGH BELOW CODE
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01),
#               metrics=['accuracy'])

# class AccuracyHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.acc = []

#     def on_epoch_end(self, batch, logs={}):
#         self.acc.append(logs.get('acc'))

# history = AccuracyHistory()

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test),
#           callbacks=[history])
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

