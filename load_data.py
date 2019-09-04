import os
import pandas as pd
import numpy as np
import h5py


def load_ecg_static_clinical_discretizer_data(label_data_folder='/data/fyj/mimic/sepsis/mortality-projects/data/short-term-mortality/24-48',
                                             static_data_folder = '/data/fyj/mimic/sepsis/mortality-projects/data/short-term-mortality/24-48',
                                             clinical_data_folder='/data/fyj/mimic/sepsis/mortality-projects/data/short-term-mortality/24-48',
                                             matched_ecg_folder='/data/fyj/mimic/sepsis/mortality-projects/data/short-term-mortality/24-48/ecg-clinical-matched'):

    labels = pd.read_csv(os.path.join(label_data_folder, 'labels.csv'), header=0, index_col=0)
    labels_dict = dict(zip(list(labels.index), list(labels.label)))
    
    static_h5file = os.path.join(static_data_folder, 'static_h5file.h5py')
    with h5py.File(static_h5file, 'r') as h5f:
        static_icuids = list(h5f.keys())

    clinical_h5file = os.path.join(clinical_data_folder, 'clinical_discretizer_2h_h5file.h5py')
    with h5py.File(clinical_h5file, 'r') as h5f:
        clinical_icuids = list(h5f.keys())

    ecg_h5file = os.path.join(matched_ecg_folder, 'subperiod_II_h5file.h5py')
    with h5py.File(ecg_h5file, 'r') as h5f:
        processed_matched_icuids = list(h5f.keys())

    labels_icuids = [str(ID) for ID in list(labels.index)]
    matched_icuids = sorted(list(set(list(labels_icuids)).intersection(set(processed_matched_icuids))))
    
    print ('ecg',len(processed_matched_icuids), 'labels',len(labels_icuids), 'matched_icuid',len(matched_icuids),'clinical',len(clinical_icuids), 'static', len(static_icuids))
    return matched_icuids, ecg_h5file, static_h5file, clinical_h5file, labels_dict

# training and tesing splition
def kfold_split(mortality_icuids, mortality_labels, folds=5, seed=1234567):
    """
    input: mortality_icuids, list of icustay_id for mortality prediction, [n,]
           mortality_labels, list of label for mortality prediction, [n,]

    return: None

    pkl.dumps [icuid_partition, label_partition]
        icuid_partition, partition of training, validation and testing icustay_id list
        label_partition, partition of training, validation and testing mortality label list
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    import copy

    mortality_icuids = np.array(copy.deepcopy(mortality_icuids))
    mortality_labels = np.array(copy.deepcopy(mortality_labels))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold = 0
    fold_icuid_label_partitions = []
    # stratifiedKFold need label for consistant distribution
    for train_index, test_index in skf.split(mortality_icuids,
                                             mortality_labels):
        train_index, val_index = train_test_split(train_index, test_size=0.1, shuffle=False, random_state=seed)
        print ('Fold ', fold)
        #print ('train:', train_index, 'val:', val_index, 'test:', test_index)
        print ('train n_samples:', len(train_index), 'val n_samples:', len(val_index), 'test n_samples:', len(test_index), '\n')

        # store icuids and labels into dictionary
        icuid_partition = {'train': list(mortality_icuids[train_index]),
                           'val': list(mortality_icuids[val_index]),
                           'test': list(mortality_icuids[test_index])}

        label_partition = {'train': list(mortality_labels[train_index]),
                           'val': list(mortality_labels[val_index]),
                           'test': list(mortality_labels[test_index])}

        fold_icuid_label_partitions.append((icuid_partition, label_partition))

        fold += 1
    return fold_icuid_label_partitions

def calculate_mean(icuid_train, h5file):
    import h5py
    for idx, icuid in enumerate(icuid_train):
        with h5py.File(h5file, 'r') as h5f:
            x = h5f[str(icuid)][:]
        if idx == 0:
            x_sum = np.zeros_like(x)
            #print (x.shape)
        x_sum += x
    x_mean = x_sum / float(len(icuid_train))
    print ('mean of training dataset', x_mean.shape)#, '\n', x_mean)
    return x_mean

def add_common_arguments(parser):
    """ Add all the parameters which are common across the tasks
    """
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--layer_end', type=int, default=20, help='end of last layer')
    parser.add_argument('--mode', type=str, default='train', help='mode: train or test')
    parser.add_argument('--weights', type=str, help='pretrained weigths of models',default=None)
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer', choices=['adam', 'rmsprop', 'sgd'])
    parser.add_argument('--loss', type=str, default='categorical_crossentropy', help='loss function')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='number of chunks to train')
    parser.add_argument('--nperseg', type=int, default=64)
    parser.add_argument('--noverlap', type=int, default=32)
    parser.add_argument('--fs', type=int, default=125, help='sampling frequency of ecg')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
    parser.add_argument('--verbose', type=int, default=2)

