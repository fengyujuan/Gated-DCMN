#help_function.py
import os
import random
import numpy as np
import pandas as pd
import h5py
import traceback
from collections import Counter

from scipy import signal
import wfdb

from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import precision_recall_curve, average_precision_score


# plot
import matplotlib.pyplot as plt


def mkdirs(filepath):
    if os.path.exists(filepath):
        print('{} exist!'.format(filepath))
        #import shutil
        #shutil.rmtree(filepath)
    else:
        filepath = filepath.rstrip('/')
        os.makedirs(filepath)
        print ('Path {} is created successful!'.format(filepath))


def norm_float(self, data):
    """For image normalization"""
    scaled = (data - np.mean(data))/ np.std(data)
    return scaled


def extend_ts(ts, max_len):
    """pad sequence to fix length
    input: ts, a list
    output: extended_ts, a list
    """
    a = int(int(max_len) / int(len(ts)))
    b = int(max_len) % int(len(ts))
    extended_ts = list(ts) * a
    extended_ts.extend(ts[0:b])
    extended_ts = list(extended_ts)
    return extended_ts


# Helper functions needed for data augmentation
def stretch_squeeze(source, length):
    import scipy.interpolate
    target = np.zeros([1, length])
    interpol_obj = scipy.interpolate.interp1d(np.arange(source.size), source)
    grid = np.linspace(0, source.size - 1, target.size)
    result = interpol_obj(grid)
    return result


# Data augmentation scheme: Random resampling
def random_resample(signals, upscale_factor = 1):
    [n_signals,length] = signals.shape
    # pulse variation from 60 bpm to 120 bpm, expected 80 bpm
    new_length = np.random.randint(
        low=int(length*80/120),
        high=int(length*80/60),
        size=[n_signals, upscale_factor])
    #print (new_length)
    signals = [np.array(s) for s in signals.tolist()]
    new_length = [np.array(nl) for nl in new_length.tolist()]
    #print (new_length)
    sigs = [stretch_squeeze(s,l) for s,nl in zip(signals,new_length) for l in nl]
    #print (sigs)
    sigs = [extend_ts(list(s), length) for s in sigs]
    #print (sigs)
    sigs = np.array(sigs)
    return sigs


# Convert ecgs into spectrogram
def convert_spectrogram(data, nperseg=64, noverlap=32, fs=125, log_spectrogram = True):
    """
    input:
    data, a list of 1-D signal

    """
    from scipy import signal
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.transpose(Sxx,[0,2,1])
    if log_spectrogram:
        Sxx = abs(Sxx) # Make sure, all values are positive before taking log
        mask = Sxx > 0 # We dont want to take the log of zero
        Sxx[mask] = np.log(Sxx[mask])
    return f, t, Sxx


def preprocess(ecg, fs=125):
    """Parameters,
    filter_types='bandpass',
    filter_order=0.3
    filter_frequency=[3, 45]
    order = int(filter_order * fs) #filter_order=0.3
    """
    import biosppy
#     plt.figure(figsize=(12,6))
#     ###step 1: normalization###
    normed_ecg = biosppy.tools.normalize(np.ravel(ecg))
    ecg = normed_ecg[0] ### if all of values are the same, it will return a list of which values are NAN
    print('normed signal',len(ecg))
    print ('normalized ecg\n', ecg)

#     plt.plot(ecg, label='normed')

    ###step 2: median smoother to remove baseline wandering###
    print ('Median filter to remove baseline wandering')
    smoothed_ecg = biosppy.tools.smoother(np.ravel(ecg), kernel='median', mirror=False, size=3)
    ecg = smoothed_ecg[0]
    print('Smoothered signal',len(ecg))
    print (ecg)
    #plt.plot(ecg, label='smoothed')

    ###step 3: filter the nosiy or artifact ###
    filter_types = ["FIR", "butter", "cheby1", "cheby2", "ellip", "bessel"]
    band_types = ['lowpass', 'highpass','bandpass', 'bandstop']

    #butterworth filter with cutoff [3-45]
    print ('FIR filter with cutoff [3-45]')
    order = int(0.3 * fs)

    """there may be ValueError if the length of the input vector x  is too short which must be at least padlen, which is 111"""
    filtered, _, _ = biosppy.tools.filter_signal(signal=np.ravel(ecg),
                                                ftype='FIR',
                                                band='bandpass',
                                                order=order, #0.08*125=10, 10-order
                                                frequency=[3,45],
                                                sampling_rate=fs)
#     ecg = filtered


#     ## butterworth filter with a wider cutoff [1-62]

#     order = int(0.3 * fs)
#     filtered, _, _ = biosppy.tools.filter_signal(signal=np.ravel(ecg),
#                                                 ftype='FIR',
#                                                 band='bandpass',
#                                                 order=order, #0.08*125=10, 10-order
#                                                 frequency=[1,60],
#                                                 sampling_rate=fs)
    print ('Butterworth low filter with cutoff 30Hz')
    ecg = filtered

    filtered, _, _ = biosppy.tools.filter_signal(signal=np.ravel(ecg),
                                                ftype='butter',
                                                band='lowpass',
                                                order=10,
                                                frequency=30,
                                                sampling_rate=fs)

#     ecg = filtered

#             filtered, _, _ = biosppy.tools.filter_signal(signal=np.ravel(ecg),
#                                                         ftype='butter',
#                                                         band='highpass',
#                                                         order=10,
#                                                         frequency=0.5,
#                                                         sampling_rate=fs)


    return filtered


def save_h5(h5f, data, target):
    shape_list = list(data.shape)
    if not h5f.__contains__(target):
        shape_list[0] = None
        dataset = h5f.create_dataset(target, data.shape, maxshape=tuple(shape_list), chunks=True)
        dataset[0:data.shape[0]] = data
        return
    else:
        dataset = h5f[target]
        len_old = dataset.shape[0]
        len_new = data.shape[0] + len_old
        shape_list[0] = len_new
        dataset.resize(tuple(shape_list))
        dataset[len_old:len_new] = data
    #h5f.close() #close cannot be in iteration


def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}

def signalshow_batch(X, y, batch_idx):
    batch_labels = ['Class label:' + str(np.argmax(y[idx,])) for idx in batch_idx]

    fig, ax = plt.subplots(1, len(batch_idx), figsize=(17, 1))

    for i, idx in enumerate(batch_idx):
        ax[i].plot(X[idx, :, 0])  # (batchsize, sequence_length, 1)
        ax[i].set_title(batch_labels[i], fontsize=10)
    # plt.show()
    return fig


def imshow_batch(X, y, batch_idx):
    batch_labels = ['Class label:' + str(np.argmax(y[idx,])) for idx in batch_idx]

    fig, ax = plt.subplots(1, len(batch_idx), figsize=(17, 1))

    for i, idx in enumerate(batch_idx):
        ax[i].imshow(X[idx, :, 0].transpose(), cmap='jet', aspect='auto')
        ax[i].grid(False)
        ax[i].axis('off')
        ax[i].invert_yaxis()
        ax[i].set_title(batch_labels[i], fontsize=10)

    # plt.show()
    return fig


def show_batch_data(generator, figs_path, to_image=False, batchsize=24, size=10):
    for i, batch in enumerate(generator):
        batch_IDs = generator.batch_IDs
        batch_ID = batch_IDs[i]
        #print ('batch_IDs\n', batch_IDs, 'length', len(batch_IDs))
        #print ('batch_ID\n', batch_ID, 'length', len(batch_ID))

        # print ('batch y:\n', y)
        # batch_idx = np.random.randint(0, batchsize, size = 10) #randomly selected number of size X and y in each batch to show.
        # batch_idx = np.arange(0, size) #selected first number of size X and y in each batch to show.
        """To firstly show at most number of #size positive samples, if not enougn, show negtive samples """
        batch_idx = []
        pos_idx = []
        neg_idx = []

        y = batch[1]
        for idx, label_encode in enumerate(y):
            if np.argmax(label_encode) == 1:
                pos_idx.append(idx)
            else:
                neg_idx.append(idx)

        if len(pos_idx) < size:
            batch_idx.extend(pos_idx)
            batch_idx.extend(neg_idx[0:size-len(pos_idx)])
        else:
            batch_idx.extend(pos_idx[0:size])

        if to_image == True:
            X = batch[0]
            if len(X) == 2:
                ecg_X = X[0]
                clinical_X = X[1]
                #print ('ecg X shape', ecg_X.shape)
                #print ('clinical X shape', clinical_X.shape)
                fig_ecg = imshow_batch(ecg_X, y, batch_idx)
                plt.savefig(os.path.join(figs_path, 'batch_ecg_{}.jpg'.format(i)))
                #fig_clinical = imshow_batch(clinical_X, y, batch_idx)
                #plt.savefig(os.path.join(figs_path, 'batch_clinical_{}.jpg'.format(i)))
            else:
                #print('X shape:', X.shape)
                #print('y shape:', y.shape)
                fig = imshow_batch(X, y, batch_idx)
                plt.savefig(os.path.join(figs_path, 'batch_{}.jpg'.format(i)))
        else:
            X = batch[0]
            if len(X) == 2:
                ecg_X = X[0]
                clinical_X = X[1]
                #print ('ecg X shape', ecg_X.shape)
                #print ('clinical X shape', clinical_X.shape)
                fig_ecg = imshow_batch(ecg_X, y, batch_idx)
                plt.savefig(os.path.join(figs_path, 'batch_ecg_{}.jpg'.format(i)))
                #fig_clinical = imshow_batch(clinical_X, y, batch_idx)
                #plt.savefig(os.path.join(figs_path, 'batch_clinical_{}.jpg'.format(i)))
            else:
                #print('X shape:', X.shape)
                #print('y shape:', y.shape)
                fig = signalshow_batch(X, y, batch_idx)
                plt.savefig(os.path.join(figs_path, 'batch_{}.jpg'.format(i)))
        if i == 5:
            break

def metric_evaluate(y_true, y_pred, model_name):

    F1_score = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall_score_ = recall_score(y_true, y_pred, average='macro')
    cohen_kappa_score_ = cohen_kappa_score(y_true, y_pred)
    
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    clf_report = classification_report(y_true, y_pred)
    report = np.array(precision_recall_fscore_support(y_true, y_pred))
    class1_report = report[:,1]
    class1_metric = list(class1_report[:-1])
    
    metrics = [] 

    def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    # Plot non-normalized confusion matrix

    class_names=['0','1']
    plot_confusion_matrix(confmat, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig(model_name)
    #plt.show()
    #plt.close()
    #print('f1_score:',F1_score,'ACC_score:',acc,'recall:',recall_score_)
    #print ('cohen kappa score:', cohen_kappa_score_)
    print('\n----class report ---:\n',clf_report)
    
    metrics.extend([F1_score, acc, recall_score_, cohen_kappa_score_])
    metrics.extend(class1_metric)
    columns = ['f1-score','acc','recall','cohen_kappa_score','precision-c1', 'recall-c1', 'f-score-c1']
    metrics = dict(zip(columns, metrics))
    return metrics



def auc_plot(y_test, y_score, model_name, n_classes=2):
    """default binary classification
    input: y_test, (n_samples, n_classes), onehot encoder
           y_score, (n_samples, n_classes)

    """
    import seaborn as sns
    sns.set()
    # Plot linewidth.
    lw = 2
    #Binary Classification
    i=1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    plt.plot(fpr[i], tpr[i], color='darkorange', lw=lw,
                 label='ROC curve of class {0} (area = {1:0.3f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(model_name)
    #print ('AUC:',roc_auc[i])
    return roc_auc[i]


def prc_plot(y_test, y_score, model_name, n_classes=2):
    """default binary classification
    input: y_test, (n_samples, n_classes), onehot encoder
    y_score, (n_samples, n_classes)"""
    # Binary Classification
    import seaborn as sns
    sns.set()
    lw = 2
    i = 1
    precisions, recalls, _ = precision_recall_curve(y_test[:, i], y_score[:, i])
    au_prc = auc(recalls, precisions)

    plt.figure()
    plt.plot(recalls, precisions, color='cornflowerblue', lw=lw,
             label='PRC curve of class {0} (area = {1:0.3f})'
                 ''.format(i, au_prc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(model_name)

    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    average_precision = average_precision_score(y_test[:, i], y_score[:, i])

    #print ('PRC: ', average_precision)
    #print("AUC of PRC = {}".format(au_prc))
    #print("min(+P, Se) = {}".format(minpse))
    return average_precision, au_prc, minpse
