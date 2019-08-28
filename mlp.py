import os

import matplotlib.pyplot as plt
from keras import backend as K
import keras
from keras.layers import Lambda, Input, Dense
from keras.models import Model
import tensorflow as tf
from keras.utils import plot_model
import sklearn
import numpy as np
import pandas as pd

from custom_callback import PrintRatioCallback
from dataset.arrhythmia_dataset import ArrhythmiaDataSet

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
arrhythmia = ArrhythmiaDataSet()
X, labels = arrhythmia.load_dataSet(representation_size=256, create=False)
list_labels = []
for i in range(len(labels)):
    list_labels.append(labels[i][0])

x_test, y_test = arrhythmia.get_anomaly()
y_train = keras.utils.to_categorical(list_labels, 2)
y_test = keras.utils.to_categorical(y_test, 2)

print(x_test.shape)
# Network parameters
feature_size = X.shape[1]
input_shape = (feature_size,)
batch_size = 1
epochs = 30

inputs = Input(shape=input_shape, name="input_layer")
h1 = Dense(64, activation='relu', name="hidden_layer1")(inputs)
h2 = Dense(32, activation='relu', name="hidden_layer2")(h1)
outputs = Dense(2, activation='softmax', name="output")(h2)

model = Model(inputs, outputs, name="multi_layer")

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
model.compile(optimizer='adam', loss=keras.losses.mean_squared_error, metrics=['accuracy'])
history = model.fit(X, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                    callbacks=[tensorboard_cb])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0] * 100, "%")
print('Test accuracy:', score[1] * 100, "%")

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
