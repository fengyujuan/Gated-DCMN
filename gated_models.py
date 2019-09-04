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

class Gated_DCMN_Layer(Layer):
    def __init__(self, edim=50, nhop=2, **kwargs):
        self.edim = edim
        self.output_dim = edim
        self.nhop = nhop
        super(Gated_DCMN_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        w_input_shape, c_input_shape, w_out_query_shape, c_out_query_shape = input_shape
        print ('waveform memory shape:',w_input_shape,'clinical memory shape:', c_input_shape)
        print ('waveform output query shape:', w_out_query_shape, 'clincial output query shape:', c_out_query_shape)

        # initialization of combination weight matrix for finally output of two memory output
        U_W = np.random.normal(0, 1, size=(self.edim, self.output_dim))
        self.U_W = K.variable(U_W)

        U_C = np.random.normal(0, 1, size=(self.edim, self.output_dim))
        self.U_C = K.variable(U_C)


        # embedding matrix initialization embedding matrix for waveform memory and clinical memory
        E_W = np.random.normal(0, 1, size=(w_input_shape[-1], self.edim))
        self.E_W = K.variable(E_W)
        print ('self.E_W', self.E_W.shape)

        E_C = np.random.normal(0, 1, size=(c_input_shape[-1], self.edim))
        self.E_C = K.variable(E_C)
        print ('self.E_C', self.E_C.shape)

        self.trainable_weights = [self.U_W, self.U_C, self.E_C, self.E_W]

        if self.nhop > 0:

           F_W = np.random.normal(0, 1, size=(w_input_shape[-1], self.edim))
           self.F_W = K.variable(F_W)
           print ('self.F_W', self.F_W.shape)

           F_C = np.random.normal(0, 1, size=(c_input_shape[-1], self.edim))
           self.F_C = K.variable(F_C)
           print ('self.F_C', self.F_C.shape)

           # initialization of weight matrix for output of each memory
           H_W = np.random.normal(0, 1, size=(self.edim, self.edim))
           self.H_W = K.variable(H_W)
           print ('self.H_W', self.H_W.shape)

           H_C = np.random.normal(0, 1, size=(self.edim, self.edim))
           self.H_C = K.variable(H_C)
           print ('self.H_C', self.H_C.shape)

           # initialization of Gated weight matrix for inteaction of output of two memory cores
           G_W = np.random.normal(0, 1, size=(self.edim, self.edim))
           self.G_W = K.variable(G_W)
           print ('self.G_W', self.G_W.shape)

           G_C = np.random.normal(0, 1, size=(self.edim, self.edim))
           self.G_C = K.variable(G_C)
           print ('self.G_C', self.G_C.shape)

           b_W = np.random.normal(0, 1, size=(1, self.edim)) # (self.batchsize, 1, self.edim)
           self.b_W = K.variable(b_W)
           print ('self.b_W', self.b_W.shape)

           b_C = np.random.normal(0, 1, size=(1, self.edim)) # (self.batchsize, 1, self.edim)
           self.b_C = K.variable(b_C)
           print ('self.b_C', self.b_C.shape)

           #self.trainable_weights.extend([self.F_W, self.F_C, self.H_W, self.H_C, self.G_W, self.G_C, self.b_W, self.b_C])
           self.trainable_weights.extend([self.F_W, self.F_C, self.G_W, self.G_C, self.b_W, self.b_C])

        super(Gated_DCMN_Layer, self).build(input_shape)  # be sure to call this at the end

    def call(self, inputs):
        assert isinstance(inputs, list)
        wm_input, cm_input, wm_out_query, cm_out_query = inputs  # (self.batchsize, self.timesteps, self.input_dim)

        # initialization for the first query for waveform memory
        wm_in = cm_out_query # (self.batchsize, 1, self.edim), optional: cm_embedding = K.dot(cm_input, self.E_C), wm_in = K.mean(cm_embedding, axis=1, keepdims=True) 

        # initializtion for the first query for clinical memory
        cm_in = wm_out_query # (self.batchsize, 1, self.edim), optional: wm_embedding = K.dot(wm_input, self.E_W), cm_in = K.mean(wm_embedding, axis=1, keepdims=True) 

        self.hop_in = []
        self.hop_in.append([wm_in, cm_in]) # (self.batchsize, 1, self.edim)

        for h in range(self.nhop):
            print ('hop ', h)
            ##############################################
            print ('waveform query', self.hop_in[-1][0].shape)  # (self.batchsize, 1, self.edim)
            wm_embedding_E = K.dot(inputs[0], self.E_W) #(self.batchsize, self.timesteps, self.edim)
            wm_embedding_F = K.dot(inputs[0], self.F_W) #(self.batchsize, self.timesteps, self.edim)
            print ('waveform sentences E', wm_embedding_E.shape)  # (self.batchsize, self.timesteps, self.edim)
            print ('waveform sentences F', wm_embedding_F.shape)  # (self.batchsize, self.timesteps, self.edim)

            wm_out = K.batch_dot(self.hop_in[-1][0],
                                 K.permute_dimensions(wm_embedding_E, (0, 2, 1)))  # (self.batchsize, 1, self.timesteps)
            wm_prob = K.softmax(wm_out)  # (self.batchsize, 1, self.timesteps)
            print ('waveform attention', wm_prob.shape)
            wm_contex = K.batch_dot(wm_prob, wm_embedding_F)  # (self.batchsize, 1, self.edim)
            print ('waveform contex', wm_contex.shape)
            print ('self.H_W', self.H_W.shape)
            
            # gated wm_dout, optional linear out: wm_dout = wm_contex + K.dot(self.hop_in[-1][0], self.H_W)  # (self.batchsize, 1, self.edim)
            wm_gated_prob = K.sigmoid(K.dot(wm_in, self.G_W) + self.b_W)
            wm_dout = wm_contex * wm_gated_prob + wm_in * (1 - wm_gated_prob)
            print('wm_dout', wm_dout.shape)

            ###############################################
            print ('clinical query', self.hop_in[-1][1].shape)  # (self.batchsize, 1, self.edim)
            cm_embedding_E = K.dot(inputs[1], self.E_C) #(self.batchsize, self.timesteps, self.edim)
            cm_embedding_F = K.dot(inputs[1], self.F_C) #(self.batchsize, self.timesteps, self.edim)
            print ('clinical sentences E', cm_embedding_E.shape)  # (self.batchsize, self.timesteps, self.edim)
            print ('clinical sentences F', cm_embedding_F.shape)  # (self.batchsize, self.timesteps, self.edim)

            cm_out = K.batch_dot(self.hop_in[-1][1],
                                 K.permute_dimensions(cm_embedding_E, (0, 2, 1)))  # (self.batchsize, 1, self.timesteps)
            cm_prob = K.softmax(cm_out)  # (self.batchsize, 1, self.timesteps)
            print ('clinical attention', cm_prob.shape)
            cm_contex = K.batch_dot(cm_prob, cm_embedding_F)  # (self.batchsize, 1, self.edim)
            print ('clinical contex', cm_contex.shape)
            print ('self.H_C', self.H_C.shape)

            # gated cm_dout, optional linear out: cm_dout = cm_contex + K.dot(self.hop_in[-1][1], self.H_C)  # (self.batchsize, 1, self.edim)
            cm_gated_prob = K.sigmoid(K.dot(cm_in, self.G_C) + self.b_C)
            cm_dout = cm_contex * cm_gated_prob + cm_in * (1 - cm_gated_prob)
            print('cm_out', cm_dout.shape)

            """
            # Gated Double-core Memory Interaction
            wm_gated_prob = K.sigmoid(K.dot(wm_dout, self.GO_W) + self.bO_W)
            cm_gated_prob = K.sigmoid(K.dot(cm_dout, self.GO_C) + self.bO_C)

            wm_in = wm_gated_prob * cm_dout + (1 - wm_gated_prob) * wm_dout
            cm_in = cm_gated_prob * wm_dout + (1 - cm_gated_prob) * cm_dout
            """
            wm_in = cm_dout
            cm_in = wm_dout
            print ('waveform query in:', wm_in.shape)
            print ('clinical query in:', cm_in.shape)

            self.hop_in.append([wm_in, cm_in])

        print ('self.U_W', self.U_W.shape)
        print ('self.U_C', self.U_C.shape)
        output = K.dot(self.hop_in[-1][0], self.U_W) + K.dot(self.hop_in[-1][1],
                                                             self.U_C)  # (self.batchsize, 1, self.output_dim)
        print ('output', output.shape)
        output = K.reshape(output, (-1, self.output_dim))  # (self.batchsize, self.output_dim)
        print ('output', output.shape)
        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        w_input_shape, c_input_shape, _, _ = input_shape
        return (w_input_shape[0], self.output_dim)

def gated_dcmn_model(clinical_dim=(12, 76),
                          static_dim=(139,),
                          spectrogram_dim=(1170, 33),
                          dropout=0.3,
                          layer_end=76,
                          edim=50,
                          n_channels=1,
                          n_classes=2,
                          nhop=2,
                          layer_filters=32,  # Start with these filters
                          filters_growth=32,  # Filter increase after each convBlock
                          strides_start=(1, 1),  # Strides at the beginning of each convBlock
                          strides_end=(2, 2),  # Strides at the end of each convBlock
                          depth=4,  # Number of convolutional layers in each convBlock
                          n_blocks=6 # Number of ConBlocks
                          ):
    # static part
    model_in_static = Input(shape=static_dim, name='static_in', dtype=np.float32)
    static_x = Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.1))(model_in_static)
    static_x = Dropout(dropout)(static_x)

    # ecg memory part
    from models import spectrogram_cnn_model
    ecg_model = spectrogram_cnn_model(spectrogram_dim=spectrogram_dim,
                                      dropout=dropout,
                                      n_classes=n_classes,
                                      n_channels=n_channels,
                                      layer_end=layer_end,
                                      layer_filters=layer_filters,  # Start with these filters
                                      filters_growth=filters_growth,  # Filter increase after each convBlock
                                      strides_start=strides_start,  # Strides at the beginning of each convBlock
                                      strides_end=strides_end,  # Strides at the end of each convBlock
                                      depth=depth,  # Number of convolutional layers in each convBlock
                                      n_blocks=n_blocks  # Number of ConBlocks
                                      )

    model_in_ecg = ecg_model.input
    ecg_x = ecg_model.get_layer(name='ecg_embedding').output
    ecg_x = BatchNormalization(center=True, scale=True)(ecg_x)
    wm_embedding_memory = LSTM(edim, return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), recurrent_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), name='wm_embedding', dropout=dropout)(ecg_x)
    # waveform out query
    wm_out_query = LSTM(edim, return_sequences=False, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), recurrent_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), name='wm_out_query', dropout=dropout)(ecg_x)

    # clinical memory part
    model_in_clinical = Input(shape=clinical_dim, name='clinical_in', dtype=np.float32)
    clinical_x = BatchNormalization(center=True, scale=True)(model_in_clinical)
    cm_embedding_memory = LSTM(edim, return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), recurrent_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), name='cm_embedding', dropout=dropout)(clinical_x)
    # clinical out query
    cm_out_query = LSTM(edim, return_sequences=False, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), recurrent_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01), name='cm_out_query', dropout=dropout)(clinical_x)

    # double-core memory network
    gated_dcmn_layer = Gated_DCMN_Layer(edim=edim, nhop=nhop)
    gated_dcmn_out = gated_dcmn_layer([wm_embedding_memory, cm_embedding_memory, wm_out_query, cm_out_query])

    # merge ecg and clinical and static
    merge_ecg_static_clinical_x = concatenate([gated_dcmn_out, static_x])
    merge_ecg_static_clinical_x = BatchNormalization(center=True, scale=True)(merge_ecg_static_clinical_x)
    merge_ecg_static_clinical_x = Dropout(dropout)(merge_ecg_static_clinical_x)
    model_out = Dense(n_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.1))(merge_ecg_static_clinical_x)

    model = Model(inputs=[model_in_ecg, model_in_static, model_in_clinical], outputs=model_out)

    model.summary()
    return model