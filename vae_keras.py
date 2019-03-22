from keras import Sequential
from keras.engine import Layer
from keras.layers import Dense, K, Input, Lambda, Layer, Multiply, Add

intermediate_dim = 128
latent_dim = 10
original_dim = 28 * 28


class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs


def negativeLogLikelihood(y_pred, y_true):
    lh = K.tf.distributions.Bernoulli(probs=y_pred)
    return - K.sum(lh.log_prob(y_true), axis=-1)


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)
z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])

eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])


decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid')
])

x_pred = decoder(z)

