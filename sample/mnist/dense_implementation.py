from keras.layers import Input, Dense
from keras.models import Model

encoding_dimension = 32
input_dimension = 28 * 28
input_image = Input(shape=(input_dimension, ))
encode = Dense(encoding_dimension, activation='relu')(input_image)
decode = Dense(input_dimension, activation='sigmoid')(encode)
autoencoder = Model(input_image, decode)

