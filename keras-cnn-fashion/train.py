from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

import wandb
from wandb.wandb_keras import WandbKerasCallback

run = wandb.init()
config = run.config

(X_train, y_train), (X_test, y_test) = mnist.load_data()

img_width=28
img_height=28

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

#reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# build model

model = Sequential()

model.add(Conv2D(config.layer_1_size, (5, 5), input_shape=(img_width, img_height,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(config.layer_2_size, (5, 5), input_shape=(img_width, img_height,1), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config.epochs, callbacks=[WandbKerasCallback()])
model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=config.epochs,
    callbacks=[WandbKerasCallback(),
                TensorBoard(write_images=True, write_grads=True, histogram_freq=1)])

model.save("convnet.h5")
