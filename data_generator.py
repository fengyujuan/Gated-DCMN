import os
import pandas as pd
import numpy as np
import h5py
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self,list_IDs=None, batchsize=24, shuffle=False):

        self.list_IDs = list_IDs
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.on_epoch_end()
        self.batch_IDs = []

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.list_IDs) / self.batchsize))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        idx_min = index * self.batchsize
        idx_max = min(idx_min + self.batchsize, len(self.list_IDs))

        indexes = self.indexes[idx_min: idx_max]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.batch_IDs = []
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        raise NotImplementedError()


class EcgStaticClinicalDataGenerator(DataGenerator):
    def __init__(self, list_IDs, ecg_h5file, static_h5file, clinical_h5file, labels,
                 train_ecg_mean=None, train_static_mean=None, train_clinical_mean=None,
                 static_dim = (139,),
                 clinical_dim=(12, 76),
                 batchsize=24, shuffle=False,
                 n_channels=1, n_classes=2,
                 sequence_length=37500, fs=125,
                 spectrogram_dim=(1170, 33), nperseg=64, noverlap=32,
                 augment=False,  extended=True, preprocessed=True,to_image=True):
        DataGenerator.__init__(self, list_IDs, batchsize, shuffle)

        self.labels = labels
        self.n_channels = n_channels
        self.n_classes = n_classes

        ## EcgGenerator initialization
        self.ecg_h5file = ecg_h5file
        self.train_ecg_mean = train_ecg_mean
        self.sequence_length = sequence_length
        self.fs = fs
        self.spectrogram_dim = spectrogram_dim
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.augment = augment
        self.extended = extended
        self.preprocessed = preprocessed
        self.to_image = to_image

        ## StaticDataGenerator initialization
        self.static_h5file = static_h5file
        self.train_static_mean = train_static_mean
        self.static_dim = static_dim

        ## ClinicalDataGenerator initialization
        self.clinical_h5file = clinical_h5file
        self.train_clinical_mean = train_clinical_mean
        self.clinical_dim = clinical_dim

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        idx_min = index * self.batchsize
        idx_max = min(idx_min + self.batchsize, len(self.list_IDs))

        indexes = self.indexes[idx_min: idx_max]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        return self.__data_generation(list_IDs_temp)

    def __data_generation(self, list_IDs_temp):
        # list of icustay_id Initialization
        IDs = []
        # label Initialization
        y = np.empty((len(list_IDs_temp)), dtype=np.int32)

        # Static Initialization
        X_static = np.empty((len(list_IDs_temp), *self.static_dim), dtype=np.float32)

        # Clinical Initialization
        X_clinical = np.empty((len(list_IDs_temp), *self.clinical_dim), dtype=np.float32)

        # ECG Initialization
        if self.to_image:
            # Initialization for 2-D image
            X_ecg = np.empty((len(list_IDs_temp), *self.spectrogram_dim, self.n_channels), dtype=np.float32)
        else:
            # Initialization for 1-D signal
            X_ecg = np.empty((len(list_IDs_temp), self.sequence_length, self.n_channels), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # as h5file is indexed by str(icustay_id)

            # Store icustay_id
            IDs.append(ID)
            # Store label
            y[i] = self.labels[int(ID)]  # assume y[i] is indexed by ID in labels

            # Store static
            with h5py.File(self.static_h5file, 'r') as h5f:
                static_1D = h5f[str(ID)][:]
            # static normalization
            normed_static_1D = static_1D - self.train_static_mean
            X_static[i, ] = normed_static_1D

            # Store Clinical Data Generation
            with h5py.File(self.clinical_h5file, 'r') as h5f:
                clinical_2D = h5f[str(ID)][:] # 2-dim，(timesteps=24, n_feats=76), clinical_2D is indexed by ID in h5file,[:] is a must to be load in memory
            # clinical normalization
            normed_clinical_2D = clinical_2D - self.train_clinical_mean
            X_clinical[i,] = normed_clinical_2D # 2-dim, (24,76)

            # ECG Data Generation
            with h5py.File(self.ecg_h5file, 'r') as h5f:
                signal_1D = h5f[str(ID)][:]  # 1-dim，(self.sequence_length, ), signal_1D is indexed by ID in h5file,[:] is a must to be load in memory

            if not self.preprocessed:
                from help_function import preprocess
                filtered_signal_1D = preprocess(np.ravel(signal_1D))
                filtered_signal_1D = pd.DataFrame(filtered_signal_1D, columns=['II'])

                filtered_signal_1D = filtered_signal_1D.dropna()
                assert len(np.ravel(filtered_signal_1D) != 0,
                           'All of values of record {} after preprocessing are NaN'.format(record_dir))

                signal_1D = filtered_signal_1D

            if not self.extended:
                from help_function import extend_ts
                extended_signal_1D = extend_ts(signal_1D, self.sequence_length)
                assert len(extended_signal_1D) == self.sequence_length
                print ('To extend raw signal with length of {} to length of {}.\n'.format(len(signal_1D),
                                                                                          len(extended_signal_1D)))
                signal_1D = extended_signal_1D

            if self.augment:
                from help_function import random_resample
                # random resampling
                signal_1D = np.array(signal_1D)
                assert len(signal_1D.shape) == 1  # (sequence_length, )
                signal_1D = np.expand_dims(signal_1D, axis=0)  # (1, sequence_length)
                signal_1D_resampled_2D = random_resample(signal_1D)  # return a (1, sequence_length)
                signal_1D = list(signal_1D_resampled_2D[0])
                # pass

            # ecg normalization
            normed_signal_1D = signal_1D - self.train_ecg_mean
            signal_1D = normed_signal_1D

            if self.to_image:
                from help_function import convert_spectrogram
                ## signal_1D must be expand to 2-D
                signal_spectrogram_3D = convert_spectrogram(np.expand_dims(signal_1D, axis=0),
                                                            nperseg=self.nperseg,
                                                            noverlap=self.noverlap,
                                                            fs=self.fs,
                                                            log_spectrogram=True)[2]  # 3-dim, (1, width, height)
                signal_spectrogram_2D = signal_spectrogram_3D[0]  # 2-dim, (width, height)
                # print ('Convert raw signal with length of {} into image of size {}.\n'.format(len(signal_1D), signal_spectrogram_2D.shape))

                X_ecg[i,] = np.expand_dims(signal_spectrogram_2D, axis=3)  # 3-D, (width,heigth, 1)

            else:
                X_ecg[i,] = np.expand_dims(signal_1D, axis=2)  # 2-dim, (self.sequence_length, 1)

        self.batch_IDs.append((IDs))


        return [X_ecg, X_static, X_clinical], keras.utils.to_categorical(y, num_classes=self.n_classes)


if __name__ == '__main__':
    print ('Class of DataGenerator.\n')
    pass