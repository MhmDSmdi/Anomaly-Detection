# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import losses
from keras.layers import Lambda, Input, Dense
from keras.losses import binary_crossentropy
from keras.models import Model

from dataset.arrhythmia_dataset import ArrhythmiaDataSet


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


arrhythmia = ArrhythmiaDataSet()
x_train, x_test = arrhythmia.load_dataSet(train_size=350, representation_size=128, create=False)
feature_size = x_train.shape[1]

# Network parameters
input_shape = (feature_size,)
intermediate_dim = 16
batch_size = 32
latent_dim = 8
epochs = 100

# Build encoder model
inputs = Input(shape=input_shape)
h1 = Dense(intermediate_dim, activation='relu')(inputs)
h2 = Dense(intermediate_dim, activation='relu')(h1)
z_mean = Dense(latent_dim)(h2)
z_log_var = Dense(latent_dim)(h2)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z])
encoder.summary()

# Build decoder model
latent_inputs = Input(shape=(latent_dim,))
h1 = Dense(intermediate_dim, activation='relu')(latent_inputs)
h2 = Dense(intermediate_dim, activation='relu')(h1)
outputs = Dense(feature_size, activation='sigmoid')(h2)
decoder = Model(latent_inputs, outputs)
decoder.summary()

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs)

# reconstruction_loss = binary_crossentropy(inputs, outputs)
# reconstruction_loss *= feature_size
# kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5
# vae_loss = K.mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)
vae.compile(optimizer='adam', loss=binary_crossentropy)
vae.summary()

vae.fit(x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

# y_test = vae.predict(x_test)
# error = losses.kullback_leibler_divergence(tf.convert_to_tensor(x_test, np.float32), y_test)
# print(error)
vae.save_weights('vae_mlp_arrhythmia.h5')
