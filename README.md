# Gated-DCMN
This repository contains Keras Code implementation of Gated Double-core Memory Network (Gated-DCMN) for multi-modal data learning in healthcare. For details about the model architecuture you can access our work: DCMN: Double Core Memory Network for PatientOutcome Prediction with Multimodal Data.


## Requirements
This package has the following requirements:
* Python 3.x`
* keras 2.2.4
* TensorFlow 1.4
* scikit-learn 0.18.2
* An NVIDIA GPU.

## Usage
To run Gated-DCMN on your data, you need to: 

(1)Change the function of loading data in load_data.py; (2) Set hyperparameters for Gated-DCMN in run_gated_dcmn.sh; (3) Run the shell script run_gated_dcmn.sh

Notes: If you want to run train_gated_dcmn.py on the publicly available Medical Information Mart for Intensive Care (MIMIC-III) database ([paper](http://www.nature.com/articles/sdata201635), [website](http://mimic.physionet.org)). Three types of features including clincial static information, clinical time-variant information and ECG signal need to be preprocessed, represented and then saved into three h5py files. All the h5file contains feature-vector indexed by ‘icustay_id’. In this script, we have prepared four inputs files from the MIMIC-III database:

*	Label information: labels.csv. Each stay is represented with a ‘int’ to indicate die in prediction window or not.
*	Clinical static information: static_h5file. Each stay is represented with a 139-dimensional vector to summarize time-invariant features including Gender, Ethnicity, Age, Los and Diagnoses. The categorical variable is one-hot encoded.
*	Clinical time-variant information: clinical_discretizer_2h_h5file: each stay is represented with a matrix of shape (24,76). We divide data collection window (DCW=24-h) into 24 timesteps with timestep-window set as 1-h. Within each timestep, we impute missing values and calculate the last value. We also add masking values to indicate whether channel is missing or not.
*	Waveform ECG information: subperiod_II_h5file: each icu stay is represented with a 37500-dimensional vector. To be noted that, this h5file only contains icustays that have both ECG signal and clinical information, that is all the icustays are ecg-clinical-matched.

### References
1. Sukhbaatar S, Weston J, Fergus R. [End-to-end memory networks](https://arxiv.org/pdf/1503.08895.pdf)
[C]//Advances in neural information processing systems. 2015: 2440-2448. 
2. Liu F, Perez J. [Gated end-to-end memory networks](https://www.aclweb.org/anthology/E17-1001)[C]//Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers. 2017: 1-10.



