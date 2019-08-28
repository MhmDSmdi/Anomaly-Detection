import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model
import tensorflow as tf
import sklearn
import keras
import numpy as np
import pandas as pd

from auc_roc import roc_callback
from dataset.arrhythmia_dataset import ArrhythmiaDataSet


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0, stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def cross_entropy(actual, predicted):
    sum_score = 0.0
    for i in range(len(actual)):
        for j in range(len(actual[i])):
            sum_score += actual[i][j] * np.log(1e-15 + predicted[i][j])
    mean_sum_score = 1.0 / len(actual) * sum_score
    return -mean_sum_score


def kl(p, q):
    return np.dot(p, q)


# Data pre processing
arrhythmia = ArrhythmiaDataSet()
X, labels = arrhythmia.load_dataSet(representation_size=128, create=False)
x_test, y_test = arrhythmia.get_anomaly()

df = pd.DataFrame()
df['label'] = list(labels)

print(X.shape)

# Network parameters
feature_size = X.shape[1]
input_shape = (feature_size,)
intermediate_dim_1 = 64
batch_size = 1
latent_dim = 32
epochs = 100

# Build encoder model
inputs = Input(shape=input_shape, name="input_layer")
h1 = Dense(intermediate_dim_1, activation='relu', name="hidden_layer1")(inputs)
z_mean = Dense(latent_dim, name="z_mean_layer", activation='linear')(h1)
z_log_var = Dense(latent_dim, name="z_log_var_layer", activation='linear')(h1)
z = Lambda(sampling, output_shape=(latent_dim,), name="sampling_layer")([z_mean, z_log_var])
encoder = Model(inputs, z_mean, name="encoder")
plot_model(encoder, to_file='encoder.png', show_shapes=True)


def vae_loss(y_true, y_predict):
    recon = K.sum(K.binary_crossentropy(y_predict, y_true), axis=1)
    kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)
    return recon + kl


# Build decoder model
decoder_hidden = Dense(intermediate_dim_1, activation='relu', name="hidden_layer2")
decoder_out = Dense(feature_size, activation='sigmoid', name="output_layer")
h1 = decoder_hidden(z)
outputs = decoder_out(h1)
d_in = Input(shape=(latent_dim,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)
plot_model(decoder, to_file='decoder.png', show_shapes=True)

# Build Variational Auto Encoder
vae = Model(inputs, outputs, name="VAE_model")
plot_model(vae, to_file='vae.png', show_shapes=True)
vae.compile(optimizer='adam', loss=keras.losses.mean_squared_error, metrics=['accuracy'])
history = vae.fit(X, X, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))
score = vae.evaluate(x_test, x_test, verbose=0)
print('Test loss:', score[0] * 100, "%")
print('Test accuracy:', score[1] * 100, "%")

vae.save("vae.h5")
# vae = load_model('vae.h5')
loss = []
labels = list(np.sort(labels))
X_index = [i for i in range(452)]
X_index = [x for _, x in sorted(zip(labels, X_index))]
ordered_X = np.array([X[index] for index in X_index])
predict_vectors = vae.predict(ordered_X)
for i in range(ordered_X.shape[0]):
    loss.append(sklearn.metrics.mean_squared_error(ordered_X[i], predict_vectors[i]))
norm = [float(i)/max(loss) for i in loss]
print('AUC:', sklearn.metrics.auc(labels, norm) * 100, '%')
