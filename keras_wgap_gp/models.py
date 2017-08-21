from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Activation
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def dense_layer(inp, f, act='relu', bn=True):
    initializer = act if act is not None else ''
    initializer = 'he_uniform' if initializer.find('relu') != -1 else 'glorot_uniform'
    out = Dense(f, use_bias=False, kernel_initializer=initializer)(inp)
    if bn: out = BatchNormalization()(out)
    
    if act == 'lrelu':
        out = LeakyReLU(alpha=0.2)(out)
    elif act is not None:
        out = Activation(act)(out)
    
    return out

def conv_layer(inp, f, k=4, s=2, p='same', act='relu', bn=True, transpose=False):
    initializer = act if act is not None else ''
    initializer = 'he_uniform' if initializer.find('relu') != -1 else 'glorot_uniform'
    fun = Conv2DTranspose if transpose else Conv2D
    out = fun(f, k, strides=s, padding=p, use_bias=False, kernel_initializer=initializer)(inp)
    if bn: out = BatchNormalization()(out)
    
    if act == 'lrelu':
        out = LeakyReLU(alpha=0.2)(out)
    elif act is not None:
        out = Activation(act)(out)
    
    return out

def make_discriminator():
    x = Input(shape=(64,64,3))
    y = conv_layer(x, 32, 3, 1, act='lrelu', bn=False)
    y = conv_layer(y, 64, act='lrelu', bn=False)
    y = conv_layer(y, 128, act='lrelu', bn=False)
    y = conv_layer(y, 256, act='lrelu', bn=False)
    y = conv_layer(y, 128, act='lrelu', bn=False)
    y = Flatten()(y)
    y = dense_layer(y, 1024, act='lrelu', bn=False)
    y = dense_layer(y, 1, act=None, bn=False)
    
    return Model(x, y)

def make_generator():
    x = Input(shape=(128,))
    y = dense_layer(x, 1024)
    y = dense_layer(x, 128*4*4)
    y = Reshape((4,4,128))(y)
    y = conv_layer(y, 256, transpose=True)
    y = conv_layer(y, 128, transpose=True)
    y = conv_layer(y, 64, transpose=True)
    y = conv_layer(y, 32, transpose=True)
    y = conv_layer(y, 32, 3, 1)
    y = conv_layer(y, 3, 3, 1, act='tanh', bn=False)
    
    return Model(x, y)
