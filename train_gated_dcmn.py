import os
import argparse
import copy
import pickle as pkl
import pandas as pd
import numpy as np
from collections import Counter
import h5py
import traceback

import random
seed = 1234567
random.seed(seed)

# function import
from load_data import load_ecg_static_clinical_discretizer_data, kfold_split, add_common_arguments
from help_function import convert_spectrogram, show_batch_data, mkdirs, auc_plot, prc_plot, metric_evaluate
from data_generator import EcgStaticClinicalDataGenerator


# plot
import matplotlib.pyplot as plt

# keras import
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard, Callback
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras.utils import plot_model

parser = argparse.ArgumentParser()
add_common_arguments(parser)
parser.add_argument('--gpu', type=int, help='GPU id', default='1')
parser.add_argument('--data', type=str, help='Path to the data source of short-term-hospital mortality task',
                    default='./data/short-term-mortality/24-48/')
parser.add_argument('--models_folder', type=str, help='Directory relative which all output files are stored',
                    default='./models/short-term-mortality/24-48')
parser.add_argument('--setting', type=str, help='setting name',
                    default='ecg-clinical-matched', choices=['ecg-clinical-matched'])
parser.add_argument('--name', type=str, help='name of training model', default='normedclassweight')
parser.add_argument('--fold', type=int, help='Fold of crossvalidation', default=0, choices=[0,1,2,3,4])
parser.add_argument('--network', type=str, required=True, default='gated_dcmn', \
                    choices=['gated_dcmn'])
parser.add_argument('--nhop', type=int, default=2, help='number of hops', choices=[1,2,3,4,5,6,7,8,9])
parser.add_argument('--edim', type=int, default=50, help='number of hidden units')
parser.add_argument('--resample', action='store_true', help='oversampling for small class')
args = parser.parse_args()
print(args)
#####################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) # '0', '1'

if 'tensorflow' == K.backend():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    tf.set_random_seed(seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    #config.gpu_options.visible_device_list = '1'
    K.set_session(tf.Session(config=config))
####################################

# folder define
setting_folder = os.path.join(args.models_folder, args.setting)
sub_dirs = ['p00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09']

###################################

mortality_icuids, ecg_h5file, static_h5file, clinical_h5file, labels_dict = load_ecg_static_clinical_discretizer_data(label_data_folder=args.data,
  static_data_folder = args.data,
  clinical_data_folder=args.data,
  matched_ecg_folder=os.path.join(args.data,args.setting))

mortality_labels = [labels_dict[int(icuid)] for icuid in mortality_icuids]

"""Crossvalidation with 5-fold stratifiedKFold,
first we will train and test our model with one-fold dataset
"""

if os.path.exists(os.path.join(setting_folder, 'fold_icu_label_partitions.pkl')):
    print ('loading five-fold crossvalidation data')
    fold_icuid_label_partitions = pkl.load(open(os.path.join(setting_folder, 'fold_icu_label_partitions.pkl'), 'rb'))
else:
    print ('Five-fold Crossvalidation spliting...')
    mkdirs(setting_folder)
    fold_icuid_label_partitions = kfold_split(mortality_icuids, mortality_labels, folds=5, seed=seed)
    pkl.dump(fold_icuid_label_partitions, open(os.path.join(setting_folder, 'fold_icu_label_partitions.pkl'), 'wb'))


# fetch one-fold dataset_partion to build and evaluate our model
fold = args.fold
icuid_partition, label_partition = fold_icuid_label_partitions[fold]
#print ('icuid_partition\n', icuid_partition)
#print ('label_partition\n', label_partition)

print ('Using Fold-{} dataset for model building and evaluation'.format(fold))
print ('training label distribution:', sorted(Counter(label_partition['train']).items(), key=lambda x: x[0]))
print ('validation label distribution:', sorted(Counter(label_partition['val']).items(), key=lambda x: x[0]))
print ('testing label distribution:', sorted(Counter(label_partition['test']).items(), key=lambda x: x[0]))

#####################################
# calculate mean of training dataset
from load_data import calculate_mean
train_ecg_mean = calculate_mean(icuid_partition['train'], ecg_h5file)
train_static_mean = calculate_mean(icuid_partition['train'], static_h5file)
train_clinical_mean = calculate_mean(icuid_partition['train'], clinical_h5file)


################################
# oversampling for class-imbalance
if args.resample:
   print ('oversampling of training dataset for class-imbalance data')
   import copy
   icuid_train = copy.deepcopy(icuid_partition['train'])
   label_train = copy.deepcopy(label_partition['train'])

   pos_icuid = [icuid for (icuid, label) in zip(icuid_train, label_train) if label==1]
   neg_icuid = [icuid for (icuid, label) in zip(icuid_train, label_train) if label==0]
   ratio = 0.5 # ratio = #pos / #neg
   resample_size = int(len(neg_icuid) * ratio)
   resample_pos_icuid = np.random.choice(pos_icuid, resample_size)

   resample_icuid_train = []
   resample_icuid_train.extend(resample_pos_icuid)
   resample_icuid_train.extend(neg_icuid)
   np.random.shuffle(resample_icuid_train)

   icuid_partition['train'] = resample_icuid_train
   label_partition['train'] = [labels_dict[int(icuid)] for icuid in resample_icuid_train]

   print ('neg_icuid:',len(neg_icuid), 'pos_icuid:', len(pos_icuid), 'train_icuid:', len(icuid_train))
   print ('neg_icuid:',len(neg_icuid), 'resample_pos_icuid:', len(resample_pos_icuid), 'resample_train_icuid:', len(resample_icuid_train))
   print ('training label distribution:', sorted(Counter(label_partition['train']).items(), key=lambda x: x[0]))
   print ('\n')

# parameters define

n_classes = len(Counter(mortality_labels))  # number of class in mortality prediction

# Initialization figs folder
figs_path = os.path.join(setting_folder, 'figs')
mkdirs(figs_path)

# fetch a sample static from h5fike
sample_icuid = icuid_partition['train'][0]

with h5py.File(static_h5file, 'r') as h5f:
    sample_static_1D = h5f[str(sample_icuid)][:]
assert sample_static_1D.shape == (139, )
static_dim = sample_static_1D.shape

# fetch a sample 2D-clinical from h5file
with h5py.File(clinical_h5file, 'r') as h5f:
    sample_clinical_2D = h5f[str(sample_icuid)][:]  # to fetch a sample from h5py

assert sample_clinical_2D.shape == (12, 76)
clinical_dim = sample_clinical_2D.shape

# fetch a sample 1D-ecg from h5file
with h5py.File(ecg_h5file, 'r') as h5f:
    sample_ecg_1D = h5f[str(sample_icuid)][:]  # to fetch a sample from h5py

assert len(sample_ecg_1D) == 125 * 5 * 60
sequence_length = len(sample_ecg_1D)

print ('The sample icustay id:', sample_icuid)
print ('The sampling 1-D ecg, length:', len(sample_ecg_1D))#, '\n', sample_ecg_1D)
print ('The sampling 1-D static, shape:', sample_static_1D.shape)#, '\n', sample_static_1D)
print ('The sampling 2-D clinical, shape:', sample_clinical_2D.shape)#, '\n', sample_clinical_2D)

plt.figure()
plt.plot(sample_ecg_1D)
#plt.show()
plt.savefig(os.path.join(figs_path, 'sample_ecg_1D.jpg'))

sample_spectrogram_3D = convert_spectrogram(np.expand_dims(list(sample_ecg_1D), axis=0), fs=args.fs, nperseg=args.nperseg, noverlap=args.noverlap, log_spectrogram=True)[2]  # (1, width, height)
plt.figure()
plt.imshow(np.transpose(sample_spectrogram_3D[0]), aspect='auto', cmap='jet')
plt.xlabel('Time/s')
plt.ylabel('Frequency/HZ')
#plt.show()
plt.savefig(os.path.join(figs_path, 'sample_spectrogram_2D.jpg'))

sample_spectrogram_2D = sample_spectrogram_3D[0]
print ('The sampling 2-D signal spectrogram, shape:', sample_spectrogram_2D.shape)#, '\n', sample_spectrogram_2D)


assert sample_spectrogram_2D.shape == (1170, 33)
spectrogram_dim = sample_spectrogram_2D.shape



params = {'batchsize': args.batchsize,
          'n_channels': args.n_channels,
          'n_classes': n_classes,
          'static_dim': static_dim,
          'clinical_dim': clinical_dim,
          'sequence_length': sequence_length,
          'fs': args.fs,
          'nperseg': args.nperseg,
          'noverlap': args.noverlap,
          'spectrogram_dim': spectrogram_dim,
          'to_image': True,
          'shuffle': False,
          'extended': True,
          'preprocessed': True,
          'augment': False}  # total 17 parameters


###################################
print (icuid_partition['train'][0])
print (icuid_partition['val'][0])
print (icuid_partition['test'][0])

print ('shuffle training icuids before optimizing')
train_icuid = copy.deepcopy(icuid_partition['train'])
np.random.shuffle(train_icuid) # shuffle before training

print ('EcgStaticClinicalDataGenerator parameters\n', params, '\n')
train_generator = EcgStaticClinicalDataGenerator(list_IDs=train_icuid,
                                                 ecg_h5file=ecg_h5file,
                                                 static_h5file =static_h5file,
                                                 clinical_h5file=clinical_h5file,
                                                 train_ecg_mean = train_ecg_mean,
                                                 train_static_mean = train_static_mean,
                                                 train_clinical_mean = train_clinical_mean,
                                                 labels=labels_dict,
                                                 **params)

val_icuid = copy.deepcopy(icuid_partition['val'])
val_generator = EcgStaticClinicalDataGenerator(list_IDs=val_icuid,
                                               ecg_h5file=ecg_h5file,
                                               static_h5file =static_h5file,
                                               clinical_h5file=clinical_h5file,
                                               train_ecg_mean = train_ecg_mean,
                                               train_static_mean = train_static_mean,
                                               train_clinical_mean = train_clinical_mean,
                                               labels=labels_dict,
                                               **params)

test_icuid = copy.deepcopy(icuid_partition['test'])
test_generator = EcgStaticClinicalDataGenerator(list_IDs=test_icuid,
                                                ecg_h5file=ecg_h5file,
                                                static_h5file =static_h5file,
                                                clinical_h5file=clinical_h5file,
                                                train_ecg_mean = train_ecg_mean,
                                                train_static_mean = train_static_mean,
                                                train_clinical_mean = train_clinical_mean,
                                                labels=labels_dict,
                                                **params)

print ('training generator numbers:', len(train_generator))
print ('validation generator numbers:', len(val_generator))
print ('testing generator numbers:', len(test_generator))

print (train_icuid[0])
print (val_icuid[0])
print (test_icuid[0])
assert icuid_partition['train'][0]!=train_icuid[0]
assert icuid_partition['val'][0]==val_icuid[0]
assert icuid_partition['test'][0]==test_icuid[0]

#show_batch_data(test_generator, figs_path, params['to_image'], batchsize=args.batchsize, size=10)

model_path = os.path.join(setting_folder, args.network)


if args.network == 'gated_dcmn':
    gated_dcmn_params = {'clinical_dim': clinical_dim,
                              'static_dim': static_dim,
                              'spectrogram_dim': spectrogram_dim,
                              'dropout': args.dropout,
                              'layer_end':args.layer_end,
                              'edim': args.edim,
                              'nhop': args.nhop,
                              'n_channels': args.n_channels,
                              'n_classes': n_classes,
                              'layer_filters': 32,  # Start with these filters
                              'filters_growth': 32,  # Filter increase after each convBlock
                              'strides_start': (1, 1),  # Strides at the beginning of each convBlock
                              'strides_end': (2, 2),  # Strides at the end of each convBlock
                              'depth': 4,  # Number of convolutional layers in each convBlock
                              'n_blocks': 6  # Number of ConBlocks
                              }

    from gated_models import gated_dcmn_model
    model = gated_dcmn_model(**gated_dcmn_params)

if args.optimizer == 'adam':
    optimizer = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.5, amsgrad=False)

if args.optimizer == 'rmsprop':
    optimizer = RMSprop(lr=args.lr, rho=0.9, epsilon=None, decay=0.0)

if args.optimizer == 'sgd':
    optimizer = SGD(lr=args.lr, decay=1e-6, momentum=0.9, nesterov=True)


if args.loss == 'categorical_crossentropy':
    model.compile(loss=args.loss, metrics=['acc'], optimizer=optimizer)


suffix = "fold{}.bs{}_edim{}_{}_{}_{}_{}".format(args.fold, args.batchsize, args.edim, args.lr, args.dropout, args.optimizer, args.loss)
model_name = args.name + '_' + args.network + '_'+ 'nhop{}'.format(args.nhop) +'_' + suffix

output_path = os.path.join(model_path, model_name)  # to store evaluation results
mkdirs(output_path)

log_path = os.path.join(output_path, 'logging')  # to store logs of network training
if args.mode == 'train':
    print ('training models....\n')

    if os.path.exists(log_path):
        print('{} exist!'.format(log_path))
        import shutil
        shutil.rmtree(log_path)
    mkdirs(log_path)

    #plot_model(model, show_shapes=True, to_file=os.path.join(output_path, '{}_plot.png'.format(args.network)))

    checkpoint = ModelCheckpoint(filepath=os.path.join(log_path, 'weights-{epoch:03d}-{val_acc:.4f}-best.hdf5'),
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True,
                              write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)
    callbacks = [early_stopping,
                 lr_reducer,
                 checkpoint,
                 tensorboard]

    from help_function import get_class_weights
    class_weight = get_class_weights(mortality_labels, smooth_factor=0.1)

    history = model.fit_generator(generator=train_generator,
                                  class_weight=class_weight,
                                  steps_per_epoch=len(train_generator),
                                  # int(np.ceil(len(icuid_partition['train'])// args.batchsize)+1), # each epoch have 20 steps, each step with #total/#steps_per_epoch
                                  epochs=args.epochs,  # 500
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=len(val_generator),
                                  # int(np.ceil(len(icuid_partition['val'])// args.batchsize)+1),
                                  workers=1,
                                  shuffle=False)

    """To save model.fit_generator history"""
    df = pd.DataFrame(history.history)
    # df.head()
    df.to_csv(os.path.join(output_path, '{}-history.csv'.format(model_name)))

    # save_model_architecture
    print ('saving model architecture..')
    model_json = model.to_json()
    with open(os.path.join(output_path, '{}-model.json'.format(model_name)), "w") as json_file:
        json_file.write(model_json)
    
    # save_weights
    print ('saving weights...')
    model.save_weights(os.path.join(output_path, '{}-weights.h5'.format(model_name)))
    
    print ('saving whole model..')
    model.save(os.path.join(output_path,'{}.h5'.format(model_name)))

elif args.mode == 'test':
    from keras.models import load_model
    print ('testing by loading whole model..')
    print ('whole model name:', os.path.join(output_path, '{}.h5'.format(model_name)))
    if args.network == 'gated_dcmn':
        from models import Gated_DCMN_Layer
        model = load_model(os.path.join(output_path, '{}.h5'.format(model_name)), custom_objects={'Gated_DCMN_Layer': Gated_DCMN_Layer})

else:
    raise ValueError("Wrong value for args.mode")

#####################################
# loss and accuracy evaluation generator
eval_res = model.evaluate_generator(train_generator)
print ('training loss and accuracy:\n', eval_res) # acc
eval_res = model.evaluate_generator(val_generator)
print ('validation loss and accuracy:\n', eval_res) # acc
eval_res = model.evaluate_generator(test_generator)
print ('testing loss and accuracy:\n', eval_res) # acc


####################################
# evaluation testing data
def evaluate_generator(generator, model, args, output_path, model_name, dataset='train', n_classes=2):
    from help_function import auc_plot, prc_plot, metric_evaluate
    generator.batch_IDs = []
    y_generator = []
    for i, batch in enumerate(generator):
        X = batch[0]
        y = batch[1]
        y_generator.extend(y)

        score = model.predict(X)

        if i == 0:
            y_score = score
            batchsize = y_score.shape[0]
            print ('first batch of {} generator\n'.format(dataset), generator.batch_IDs)
        else:
            y_score = np.concatenate([y_score, score], axis=0)

        if i==len(generator)-1:
            batch_IDs = generator.batch_IDs
            print ('number of batch:', len(batch_IDs), 'batchsize:', batchsize)
            print ('shape of last batch')
            for x in X:
                print (x.shape)

            print (y.shape)

    y_generator = np.vstack(y_generator)
    # print ('y_generator\n',y_generator)
    y_generator_true = np.argmax(y_generator, axis=-1) # true label

    eval_res = model.evaluate_generator(generator)
    print ('{}ing evaluation results:\n'.format(dataset), eval_res)

    #y_score = np.vstack(y_score)
    print ('y_score\n', y_score.shape)#, '\n', y_score)
    y_pred = np.argmax(y_score, axis=-1)
    print (sorted(Counter(list(y_pred)).items(), key=lambda x: x[0]))
    #print ('y_pred\n', y_pred)

    metrics = metric_evaluate(y_generator_true, y_pred, os.path.join(output_path, '{}_{}_cmt.png'.format(dataset, args.network)))
    auc = auc_plot(y_generator, y_score, os.path.join(output_path, '{}_{}_auc.png'.format(dataset, args.network)) ,n_classes)
    metrics['auc'] = auc
    average_precision, au_prc, minpse = prc_plot(y_generator, y_score, os.path.join(output_path, '{}_{}_prc.png'.format(dataset, args.network)) ,n_classes)
    metrics['average_precision'] = average_precision
    metrics['auprc'] = au_prc
    metrics['minpse'] = minpse
    metrics = dict(sorted(metrics.items(), key=lambda x: x[0]))
    print (metrics)
    metrics_df = pd.DataFrame(list(metrics.values()), index=(metrics.keys()), columns=['metric'])
    metrics_df.to_csv(os.path.join(output_path, '{}_{}-metrics.csv'.format(dataset, model_name)))
    print ('{}ing evalution end!\n'.format(dataset))
    return y_generator_true, y_generator

print ('datasets evaluations \n')
y_train_generator_true, y_train_generator = evaluate_generator(train_generator, model, args, output_path, model_name, dataset='train', n_classes=2)
y_test_generator_true, y_test_generator = evaluate_generator(test_generator, model, args, output_path, model_name, dataset='test', n_classes=2)



