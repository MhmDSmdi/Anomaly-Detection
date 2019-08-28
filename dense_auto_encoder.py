import keras
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model

from dataset.arrhythmia_dataset import ArrhythmiaDataSet

arrhythmia = ArrhythmiaDataSet()
X, labels = arrhythmia.load_dataSet(representation_size=128, create=False)
list_labels = []
for i in range(len(labels)):
    list_labels.append(labels[i][0])

x_test, y_test = arrhythmia.get_anomaly()
y_train = keras.utils.to_categorical(list_labels, 2)
y_test = keras.utils.to_categorical(y_test, 2)

# Network parameters
feature_size = X.shape[1]
input_shape = (feature_size,)
batch_size = 1
epochs = 50

inputs = Input(shape=input_shape, name="input_layer")
h1 = Dense(128, activation='relu', name="encoder_hidden_layer1")(inputs)
# h2 = Dense(32, activation='relu', name="encoder_hidden_layer2")(h1)
# middle = Dense(16, activation='relu', name="middle_layer")(h2)
# decoder_h2 = Dense(32, activation='relu', name="decoder_hidden_layer2")(middle)
# decoder_h1 = Dense(64, activation='relu', name="decoder_hidden_layer1")(decoder_h2)
outputs = Dense(feature_size, activation='relu', name="output_layer")(h1)
model = Model(inputs, outputs, name="dense_autoencoder")

model.compile(optimizer='adam', loss=keras.losses.mean_absolute_error, metrics=['accuracy'])
history = model.fit(X, X, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))

score = model.evaluate(x_test, x_test, verbose=0)
print('Test loss:', score[0] * 100, "%")
print('Test accuracy:', score[1] * 100, "%")

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
