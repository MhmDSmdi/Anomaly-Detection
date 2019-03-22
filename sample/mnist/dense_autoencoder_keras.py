from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

encoding_dimension = 32
input_dimension = 28 * 28
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print("Train Data Dimension : {}".format(x_train.shape))
print("Test Data Dimension : {}".format(x_test.shape))

# Constructing Auto Encoder
input_image = Input(shape=(input_dimension,))
encode = Dense(encoding_dimension, activation='relu')(input_image)
decode = Dense(input_dimension, activation='sigmoid')(encode)
autoencoder = Model(input_image, decode)
encoder = Model(input_image, encode)
encoded_input = Input(shape=(encoding_dimension,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

num_digit_show = 10
plt.figure(figsize=(20, 4))
for i in range(num_digit_show):
    # display original
    ax = plt.subplot(2, num_digit_show, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, num_digit_show, i + 1 + num_digit_show)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
