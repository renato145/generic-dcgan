#import pdb
import os
import numpy as np
from time import time
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Activation, BatchNormalization, Flatten
from keras.layers import UpSampling2D, Conv2D, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K

LRELU = 0.2

def custom_conv(x, filters, kernel=3, bn=True, activation='relu', ud_sample='None'):
    if ud_sample == 'up':
        x = UpSampling2D()(x)
    
    initialization = 'he_uniform' if activation == 'relu' else 'glorot_uniform'
    out = Conv2D(filters, kernel, padding='same', kernel_initializer=initialization)(x)
    if bn:
        out = BatchNormalization()(out)
    out = Activation(activation)(out)
    
    if ud_sample == 'down':
        out = MaxPool2D()(out)
    
    return out

def custom_dense(x, units, bn=True, activation='lrelu'):
    initialization = 'he_uniform' if activation.find('relu') == -1 else 'glorot_uniform'
    activation_fn = LeakyReLU(LRELU) if activation == 'lrelu' else Activation(activation)
    out = Dense(units, kernel_initializer=initialization)(x)
    if bn:
        out = BatchNormalization()(out)
    out = activation_fn(out)
    
    return out

def generator_model(latent_dims):
    x = Input((latent_dims,))
    y = custom_dense(x, 1024)
    y = custom_dense(y, 128*7*7)
    y = Reshape((7,7,128))(y)
    y = custom_conv(y, 64, 5, ud_sample='up')
    y = custom_conv(y, 1, 5, ud_sample='up', activation='tanh', bn=False)
    model = Model(x, y)
    
    return model

def discriminator_model(input_shape=(28,28,1)):
    x = Input(input_shape)
    y = custom_conv(x, 64, 5, ud_sample='down')
    y = custom_conv(y, 128, 5, ud_sample='down')
    y = Flatten()(y)
    y = custom_dense(y, 1024)
    y = custom_dense(y, 1, bn=False, activation='sigmoid')
    model = Model(x, y)
    
    return model

def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    
    return model

class GanModel(object):
    def __init__(self, g_weights='generator.h5', d_weights='discriminator.h5', data='data.npy',
                 lr=5e-4, latent_dims=100):
        self.lr = lr
        self.latent_dims = latent_dims
        self.d_loss = []
        self.g_loss = []
        self.data = data
        self.load_data()
        self.d_weights = d_weights
        self.g_weights = g_weights
        self.d = discriminator_model()
        self.g = generator_model(latent_dims)
        self.dg = generator_containing_discriminator(self.g, self.d)
        self.g.compile(loss='binary_crossentropy', optimizer=Adam(self.lr))
        self.dg.compile(loss='binary_crossentropy', optimizer=Adam(self.lr))
        self.d.trainable = True
        self.d.compile(loss='binary_crossentropy', optimizer=Adam(self.lr))
        self.load_weights()
        self.save_data()
    
    def load_data(self):
        if os.path.exists(self.data):
            data = np.load(self.data).item()
            self.lr = data['lr']
            self.latent_dims = data['latent_dims']
            self.d_loss = data['d_loss']
            self.g_loss = data['g_loss']

    def save_data(self):
        data = {'lr': self.lr, 'latent_dims': self.latent_dims, 'd_loss': self.d_loss, 'g_loss': self.g_loss}
        np.save(self.data, data)

    def load_weights(self):
        if os.path.exists(self.g_weights) and os.path.exists(self.d_weights):
            self.d.load_weights(self.d_weights)
            self.g.load_weights(self.g_weights)
        
    def train(self, train_data, test_data=None, batch_size=128, epochs=20):
        X_train, y_train = train_data
        if test_data:
            X_test, y_test = test_data
        
        n_batches = X_train.shape[0] // batch_size
        noise = np.zeros((batch_size, self.latent_dims))
        print(f'Batches per epoch : {n_batches}')
        for epoch in range(epochs):
            t0 = time()
            print(f'Epoch : {epoch + 1:04}/{epochs:04}')
            for index in range(n_batches):
                for i in range(batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, self.latent_dims)
                    
                image_batch = X_train[index*batch_size:(index+1)*batch_size]
                generated_images = self.g.predict(noise, verbose=0)
                X = np.concatenate((image_batch, generated_images))
                y = [0.9] * batch_size + [0.0] * batch_size
                d_loss = self.d.train_on_batch(X, y)
                self.d_loss.append(d_loss)
                for i in range(batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, self.latent_dims)
                    
                self.d.trainable = False
                g_loss = self.dg.train_on_batch(noise, [1] * batch_size)
                self.d.trainable = True
                self.g_loss.append(g_loss)
                if index == 0:
                    print(f'{index+1:04}/{n_batches:04} gen loss: %.5f - disc loss: %.5f' %
                          (g_loss, d_loss))
                print(f'{index+1:04}/{n_batches:04} gen loss: %.5f - disc loss: %.5f' %
                          (g_loss, d_loss), end='\r')
                if index % 10 == 9:
                    self.g.save_weights(self.g_weights)
                    self.d.save_weights(self.d_weights)
                    self.save_data()
                
            print(f'{index+1:04}/{n_batches:04} gen loss: %.5f - disc loss: %.5f (%.2fs)' %
                  (g_loss, d_loss, time() - t0))

    def generate(self, batch_size, verbose=1):
        noise = np.zeros((batch_size, self.latent_dims))
        for i in range(batch_size):
            noise[i, :] = np.random.uniform(-1, 1, self.latent_dims)
            
        generated_images = self.g.predict(noise, verbose=verbose)
        
        return generated_images