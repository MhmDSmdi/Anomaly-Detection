import os

import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.utils import plot_model
import tensorflow as tf
import numpy as np
import pandas as pd

from dataset.arrhythmia_dataset import ArrhythmiaDataSet


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0, stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()
    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit
            plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = (n - 1) * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(filename)
        plt.show()


# Data pre processing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, [-1, 784])
x_test = np.reshape(x_test, [-1, 784])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Network parameters
feature_size = 28 * 28
input_shape = (feature_size,)
intermediate_dim_1 = 512
batch_size = 128
latent_dim = 2
epochs = 50

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

vae.compile(optimizer='adam', loss=K.binary_crossentropy, metrics=['accuracy'])
history = vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plot_results((encoder, decoder),
             (x_test, y_test),
             batch_size=batch_size,
             model_name="vae_mlp")

# vae.save_weights('vae_mlp_arrhythmia.h5')
