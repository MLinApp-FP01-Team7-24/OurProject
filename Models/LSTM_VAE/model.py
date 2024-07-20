import tensorflow as tf
from tensorflow import keras, data
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import layers, regularizers, activations
from tensorflow.keras import backend as K

class Sampling(layers.Layer):
    def __init__(self, z_dim, name='sampling_z'):
        super(Sampling, self).__init__(name=name)
        self.z_dim = z_dim

    def call(self, inputs):
        mu, logvar = inputs
        sigma = K.exp(logvar * 0.5)
        epsilon = K.random_normal(shape=(mu.shape[0], self.z_dim), mean=0.0, stddev=1.0)
        return mu + epsilon * sigma

    def get_config(self):
        config = super(Sampling, self).get_config()
        config.update({'name': self.name})
        return config

class Encoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.encoder_inputs = keras.Input(shape=(time_step, x_dim))
        self.encoder_lstm = layers.LSTM(lstm_h_dim, activation='softplus', name='encoder_lstm', stateful=True)
        self.z_mean = layers.Dense(z_dim, name='z_mean')
        self.z_logvar = layers.Dense(z_dim, name='z_log_var')
        self.z_sample = Sampling(z_dim)

    def call(self, inputs):
        self.encoder_inputs = inputs
        hidden = self.encoder_lstm(self.encoder_inputs)
        mu_z = self.z_mean(hidden)
        logvar_z = self.z_logvar(hidden)
        z = self.z_sample((mu_z, logvar_z))
        return mu_z, logvar_z, z

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'name': self.name,
            'z_sample': self.z_sample.get_config()
        })
        return config
    
class Decoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.z_inputs = layers.RepeatVector(time_step, name='repeat_vector')
        self.decoder_lstm_hidden = layers.LSTM(lstm_h_dim, activation='softplus', return_sequences=True, name='decoder_lstm')
        self.x = layers.Dense(x_dim, name='x_mean')


    def call(self, inputs):
        z = self.z_inputs(inputs)
        hidden = self.decoder_lstm_hidden(z)
        x = self.x(hidden)
        return x

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'name': self.name
        })
        return config

class LSTM_VAE(keras.Model):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='lstm_vae', **kwargs):
        super(LSTM_VAE, self).__init__(name=name, **kwargs)

        self.encoder = Encoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)
        self.decoder = Decoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)

        self.loss_metric = keras.metrics.Mean(name='loss')

    def call(self, inputs):
        mu_z, logvar_z, z = self.encoder(inputs)
        y = self.decoder(z)

        kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mu_z) + tf.exp(logvar_z) - logvar_z - 1, axis=-1))
        self.add_loss(kl_loss)

        return y

    def get_config(self):
        config = {
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'name': self.name
        }
        return config

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            y = self(x, training=True)
            loss = K.mean(K.square(x - y))
            loss += sum(self.losses)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_metric.update_state(loss)

        return {'loss': self.loss_metric.result()}

def save_model(model, model_dir):
    with open(model_dir + 'lstm_vae.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(model_dir + 'lstm_vae_ckpt')

def load_model(model_dir):
    lstm_vae_obj = {'Encoder': Encoder, 'Decoder': Decoder, 'Sampling': Sampling}
    with keras.utils.custom_object_scope(lstm_vae_obj):
        with open(model_dir + 'lstm_vae.json', 'r'):
            model = keras.models.model_from_json(model_dir + 'lstm_vae.json')
        model.load_weights(model_dir + 'lstem_vae_ckpt')
    return model