import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import matplotlib.pyplot as plt

# load mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

epochs=10
batchsize=60

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Input normalization
x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert class vectors to binary class matrices
num_classes= y_train.shape[1]

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# build model
model = Sequential()

model.add(Dense(units=512, activation='relu', input_dim=784))
model.add(Dense(units=512, activation='relu', input_dim=512,kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(units=128, activation='relu', input_dim=512,kernel_regularizer=regularizers.l2(0.04)))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes,activation='softmax',input_dim=128, kernel_regularizer=regularizers.l2(0.1)))


model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

model.summary()

# train
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batchsize, validation_data=(x_test, y_test), verbose=1)
loss_and_metrics = model.evaluate(x_test, y_test,batch_size=batchsize, verbose=0)

print('Test loss:', loss_and_metrics[0])
print('Test accuracy:', loss_and_metrics[1])

# plot accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('NN Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoc')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plot loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('NN Loss')
plt.ylabel('loss')
plt.xlabel('epoc')
plt.legend(['train', 'test'], loc='upper left')
plt.show()