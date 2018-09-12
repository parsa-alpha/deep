from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation,Dropout
import read_caltech as data_reader
import matplotlib.pyplot as plt
import tensorflow as tf

model = Sequential()

model.add(Conv2D(4,(3, 3), padding='same', input_shape=(150, 250, 3)))
model.add(Dropout(0.1))
model.add(MaxPool2D((2, 2), strides=(2, 2)))


model.add(Conv2D(6,(3, 3), padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D((2, 2), strides=(2, 2)))

model.add(Conv2D(8,(3, 3), padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D((2, 2), strides=(2, 2)))

model.add(Conv2D(16,(3, 3), padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation(tf.nn.softmax))

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

X_train, y_train = data_reader.get_data_mat('dataset/train')
X_test, y_test = data_reader.get_data_mat('dataset/test')

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

history = model.fit(X_train, y_train, batch_size=10, epochs=20, validation_data=(X_test, y_test))
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()

model.save('models\model5.h5')
