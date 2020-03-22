import keras
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import backend as K
 
 
batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
 
 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
 

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
 
#modeling
model = Sequential()
# conv1
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=input_shape))
#conv2
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
#pool3
model.add(MaxPool2D(pool_size=(2, 2)))

#pool4
model.add(MaxPool2D(pool_size=(2, 2)))
#in case overfitting
model.add(Dropout(0.25))
#full connected output128
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))

#classification，softmax activation
model.add(Dense(units=num_classes, activation='softmax'))
 
#optimizing
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
#interation
model.fit(x=x_train, y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
#evaluation
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print('Test loss: {0}'.format(score[0]))
print('Test accuracy: {0}'.format(score[1]))
