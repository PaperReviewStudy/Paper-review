import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from datetime import datetime

import os

def write_log(logs, ex):
    ex.log_scalar('loss', logs.get('loss'))
    ex.log_scalar('val_loss', logs.get('val_loss'))

def train_model(args, ex):
    img_rows = 28
    img_cols = 28

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    print(x_train.shape)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(args.hidden_dim1, kernel_size=(5, 5), strides=(1, 1), padding='same',
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(args.hidden_dim2, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(args.hidden_dim3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    sgd = SGD(lr=args.lr, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['accuracy'])

    model.summary()
    cb = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: write_log(logs, ex)
    )

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epoch,
              verbose=1,
              validation_data=(x_test, y_test))

    test_loss = model.evaluate(x_test, y_test)

    print("final %s" % test_loss)
    '''@nni.report_final_result(test_loss)'''

    return test_loss

