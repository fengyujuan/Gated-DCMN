# Gated-DCMN
This is the work of Gated Double-core Memory Network.
To train the Gated-DCMN model, we need four source data files extracted from the the MIMIC-III dataset, 
Except for labels, theree types of features is preprocessed, represented and then saved into three h5py files. All the h5file contains feature-vector indexed by ‘icustay_id’. There are:
(a)	Label information: labels.csv. Each stay is represented with a ‘int’ to indicate die in prediction window or not.
(b)	Static information: static_h5file. Each stay is represented with a 139-dimensional vector to summarize time-invariant features including Gender, Ethnicity, Age, Los and Diagnoses. The categorical variable is one-hot encoded.
(c)	Clinical information: clinical_discretizer_2h_h5file: each stay is represented with a matrix of shape (24,76). We divide data collection window (DCW=24-h) into 24 timesteps with timestep-window set as 1-h. Within each timestep, we impute missing values and calculate the last value. We also add masking values to indicate whether channel is missing or not.
(d)	ECG information: subperiod_II_h5file: each icu stay is represented with a 37500-dimensional vector. To be noted that, this h5file only contains icustays that have both ECG signal and clinical information, that is all the icustays are ecg-clinical-matched.
