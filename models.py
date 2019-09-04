import os
import numpy as np

# keras import
import keras
from keras import backend as K
from keras import layers
from keras.engine.topology import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import Dropout, BatchNormalization
from keras import regularizers


def MeanOverTime():
    lam_layer = layers.Lambda(lambda x: K.mean(x, axis=1), output_shape=lambda s: (s[0], s[-1]), name='mean_embedding')
    return lam_layer

# conv1D block
def conv1D_block(model, input_shape=(37500,1),
                 depth=4, layer_filters=32, filters_growth=32,
                 stride_start=1, stride_end=2,first_layer = False):

    conv_params = {'kernel_size':3,
                   'padding':'same',
                   'dilation_rate':1,
                   'activation':None,
                    #'data_format': 'channels_last', #may only conv2d need!
                   'kernel_initializer':'glorot_normal'}

    for l in range(depth):
        if first_layer:
            model.add(layers.Conv1D(filters = layer_filters,
                                    strides = stride_start,
                                    input_shape = input_shape,
                                    **conv_params))
            first_layer = False
        else:
            if l== depth - 1:
            # Last layer in each block is different: adding filters and using stride 2
                layer_filters += filters_growth
                model.add(layers.Conv1D(filters = layer_filters,
                                        strides = stride_end,
                                        **conv_params))
            else:
                model.add(layers.Conv1D(filters = layer_filters,
                                        strides = stride_start,
                                        **conv_params))
        # Continue with batch normalization and activation for all layers in the block
        model.add(layers.BatchNormalization(center = True, scale = True))
        model.add(layers.Activation('relu'))

    return model



# Convolutional blocks
def conv2d_block(model, input_shape=(1170, 33), depth=4,
                 layer_filters=32, filters_growth=32,
                 strides_start=(1, 1), strides_end=(2, 2),
                 first_layer=False):
    """Convolutional block.
    depth: number of convolutional layers in the block (4)
    filters: 2D kernel size (32)
    filters_growth: kernel size increase at the end of block (32)
    first_layer: provide input_shape for first layer
    """

    # Fixed parameters for convolution
    conv_parms = {'kernel_size': (3, 3),
                  'padding': 'same',
                  'dilation_rate': (1, 1),
                  'activation': None,
                  'data_format': 'channels_last',
                  'kernel_initializer': 'glorot_uniform',
                  'kernel_regularizer':regularizers.l2(0.01)}

    for l in range(depth):

        if first_layer:

            # First layer needs an input_shape
            model.add(layers.Conv2D(filters=layer_filters,
                                    strides=strides_start,
                                    input_shape=input_shape, **conv_parms))
            first_layer = False

        else:
            # All other layers will not need an input_shape parameter
            if l == depth - 1:
                # Last layer in each block is different: adding filters and using stride 2
                layer_filters += filters_growth
                model.add(layers.Conv2D(filters=layer_filters,
                                        strides=strides_end, **conv_parms))

            else:
                model.add(layers.Conv2D(filters=layer_filters,
                                        strides=strides_start, **conv_parms))

        # Continue with batch normalization and activation for all layers in the block
        model.add(layers.BatchNormalization(center=True, scale=True))
        model.add(layers.Activation('relu'))
    return model


def spectrogram_cnn_model(spectrogram_dim=(1170, 33),
                          dropout=0.3,
                          n_classes=2,
                          n_channels=1,
                          layer_end=50,
                          layer_filters=32,  # Start with these filters
                          filters_growth=32,  # Filter increase after each convBlock
                          strides_start=(1, 1),  # Strides at the beginning of each convBlock
                          strides_end=(2, 2),  # Strides at the end of each convBlock
                          depth=4,  # Number of convolutional layers in each convBlock
                          n_blocks=6 # Number of ConBlocks
                          ): #total 11

    input_shape = (*spectrogram_dim, n_channels)  # input shape for first layer
    model = Sequential()

    for block in range(n_blocks):
        # Provide input only for the first layer
        if block == 0:
            provide_input = True
        else:
            provide_input = False

        model = conv2d_block(model,
                             input_shape=input_shape,
                             depth=depth,
                             layer_filters=layer_filters,
                             filters_growth=filters_growth,
                             strides_start=strides_start,
                             strides_end=strides_end,
                             first_layer=provide_input)
        model.add(Dropout(dropout))
        # Increase the number of filters after each block
        layer_filters += filters_growth

    # Fixed parameters for 1*1 convolution
    conv_parms = {'kernel_size': (1, 1),
                  'padding': 'same',
                  'dilation_rate': (1, 1),
                  'activation': 'relu',
                  'data_format': 'channels_last',
                  'kernel_initializer': 'glorot_uniform',
                  'kernel_regularizer':regularizers.l2(0.01)}

    model.add(layers.Conv2D(filters=layer_end, strides=(1, 1), **conv_parms))
    # Remove the frequency dimension, so that the output can feed into LSTM
    # Reshape to (batch, time steps, filters)
    model.add(layers.TimeDistributed(Flatten(), name='ecg_embedding')) # (ars.batchsize, timesteps, layer_end)
    # model.add(layers.core.Masking(mask_value = 0.0)) # if padded with zero
    model.add(BatchNormalization(center=True, scale=True))
    model.add(MeanOverTime())  # Training phase 1: training CNN with MeanOverTime(), 500 epoches
    model.add(Dropout(dropout))

    # And a fully connected layer for the output
    model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(dropout))
    model.add(layers.Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout))
    model.add(layers.Dense(n_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.1)))

    model.summary()
    return model



