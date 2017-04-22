from keras.models import Model
from keras.layers import Input, Dense, Reshape, Activation, BatchNormalization, Flatten
from keras.layers import UpSampling2D, Conv2D, MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

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

def generator_model(input_shape=(100,)):
    x = Input(input_shape)
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