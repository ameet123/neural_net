import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras import losses

# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Generate dummy data
#
# data = np.random.random((1000, 100))
# labels = np.random.randint(2, size=(1000, 1))
#
# # Train
# model.fit(data, labels, epochs=10, batch_size=32)

batch_size = 128
num_classes = 10
epochs = 20
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print("digit for 1st example:{}".format(y_train[0]))
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("categorical label for 1st example:{}".format(y_train[0]))

# Model build
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss=losses.categorical_crossentropy, optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print (type(score))