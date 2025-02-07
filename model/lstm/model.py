#!/usr/bin/python3
import argparse
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from numpy import load

# Prepare arguements
parser = argparse.ArgumentParser(description="Hyperparameters for tranformer model")
parser.add_argument("--num_classes", dest="num_classes", default=499, type=int,
                    help="number of websites to be classified")
parser.add_argument("--npz_file_path", dest="npz_file_path", default="../../dataset/dataset_500_200.npz", type=str,
                    help="the data location")
args = parser.parse_args()


npz_file_path = args.npz_file_path
dict_data = load(npz_file_path)
x_train, y_train, x_test, y_test = dict_data["x_train"], dict_data["y_train"], dict_data["x_test"], dict_data["y_test"]
x_train_len = len(x_train)
x_val, y_val = x_train[int(x_train_len * 0.75):], y_train[int(x_train_len * 0.75):]
x_train, y_train = x_train[:int(x_train_len * 0.75)], y_train[:int(x_train_len * 0.75)]

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print("using gpu")
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# define model

CLASSES = args.num_classes + 1
PATIENCE=1
EPOCHS=120
BATCH_SIZE = 64
DROPOUT = 0.2
LEARNING_RATE=0.0001
REGULARIZATION = 0.001

model = tf.keras.models.Sequential([
    Input(shape=x_train[0].shape, dtype = tf.float32),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(REGULARIZATION))),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(REGULARIZATION))),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(REGULARIZATION))),
    Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(REGULARIZATION))),
    Dense(units=128, activation = 'relu', kernel_regularizer=l2(REGULARIZATION)),
    Dropout(DROPOUT),
    Dense(units=CLASSES, activation = 'softmax', kernel_regularizer=l2(REGULARIZATION)),
    Reshape([1, -1]),
])

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=PATIENCE,
                               mode='min')

model.compile(loss='sparse_categorical_crossentropy',
                   optimizer=Adam(learning_rate = LEARNING_RATE),
                   metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_data=(x_val,y_val),
                    callbacks=[early_stopping,],
                    batch_size = BATCH_SIZE)

accuracy_test = model.evaluate(x_test,y_test)
print(accuracy_test)
