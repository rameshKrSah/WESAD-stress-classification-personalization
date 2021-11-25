
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import my_utils as gb_utl

data_folder = '../Processed Data/WESAD/Segments/60 seconds 50 % overlap/Subjects/'
subject_ids = [
    'S6',
    'S14',
    'S7',
    'S3',
    'S17',
    'S8',
    'S13',
    'S4',
    'S5',
    'S16',
    'S11',
    'S2',
    'S9',
    'S10',
    'S15'
]

def combine_class_data(baseline_path, amusement_path, stressed_path, include_amusement = False):
    """
        @brief: Load the data for different stress class and return X, Y. If amusement is included into the 
            baseline class, the labels assigned to amusement is 0 same as that of baseline class. 
            
        @param: baseline_path (string): path to baseline data, amusement_path (string): path to the amusement data,
            stressed_path (string): path to stressed data, and include_amusement (Boolean): whether to include 
            amusement data into baseline or not. By default amusement data is not included into baseline.
            
        @return: X, Y : NumPy arrays.
    
    """
    stress_label = 1
    not_stress_label = 0
    
    # load the segments
    baseline_segments = gb_utl.read_data(baseline_path)
    stress_segments = gb_utl.read_data(stressed_path)
    
    # combine the baseline and stress segments
    X = np.concatenate([baseline_segments, stress_segments], axis = 0)
    Y = np.concatenate([np.zeros(baseline_segments.shape[0], dtype=int), 
                       np.ones(stress_segments.shape[0], dtype=int)
                       ])
    if include_amusement:
        amusement_segments = gb_utl.read_data(amusement_path)
        X = np.concatenate([X, amusement_segments])
        Y = np.concatenate([Y, np.zeros(amusement_segments.shape[0], dtype=int)])
    
    return X, Y

def get_eda_model():
    temp_model = keras.Sequential([
        keras.layers.Conv1D(filters = 100, kernel_size = (5), strides = 1, activation = tf.nn.relu, 
                            input_shape = (window_length, 1)),
        
        keras.layers.Conv1D(filters = 100, kernel_size = (10), strides = 1, activation = tf.nn.relu),
        keras.layers.GlobalMaxPool1D(),
        
        keras.layers.Dense(units = 128, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.3),
        
        keras.layers.Dense(units = 64, activation = tf.nn.relu, name='penultimate'),
        keras.layers.Dropout(rate = 0.2),
        keras.layers.Dense(units = n_classes, activation = tf.nn.softmax)
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, 
                       optimizer = keras.optimizers.Adam(learning_rate = LR_RATE), 
                      metrics = ['accuracy'])
    
    return temp_model


# # Leave One Subject Out Analysis Protocol
# 
# - For each subject load that subject data as test data. Load the remaining subject data as training data. 
# - Train the model on the training data, for fixed number of EPOCHs and BATCH_SIZE for each subject.
# - Evaluate the trained model on the training and testing data. Also, get the metrics for the training and test sets.
# - Store the metrics in a dictionary with the test data subject as the key.
# - Will not do k-fold cross validation.


window_length = 240 # 60 seconds window with 4Hz sampling frequency
LR_RATE = 0.001
n_classes = 2
folder_files = os.listdir(data_folder)

loso_result_dict = {}

def get_data_paths(eda_files):
    baseline_path = ""
    stress_path = ""
    amusement_path = ""
    
    for f in eda_files:
        if 'baseline' in f:
            baseline_path = data_folder + f
        elif 'stress' in f:
            stress_path = data_folder + f
        elif 'amusement' in f:
            amusement_path = data_folder + f
            
    return baseline_path, stress_path, amusement_path

for test_subject in subject_ids:
    print("Running LOSO for subject", test_subject)
    test_subject_eda_files = [r for r in folder_files if test_subject in r]
    train_subjects = [r for r in subject_ids if test_subject != r]
    
    # get the test data files path
    baseline_path, stress_path, amusement_path = get_data_paths(test_subject_eda_files)
            
    # load the test data using the data file
    x_ts, y_ts = combine_class_data(baseline_path, amusement_path, 
                                        stress_path, include_amusement=False)
    print("Test data")
    print(x_ts.shape, y_ts.shape)
    
    x_tr = np.zeros((1, window_length))
    y_tr = np.array([-1])
    
    # load the training set
    print("Training set subjects ", end="")
    for train_subject in train_subjects:
        print(train_subject, end=", ")
        train_subject_eda_files = [r for r in folder_files if train_subject in r]
        baseline_path, stress_path, amusement_path = get_data_paths(train_subject_eda_files)
        
        x_tp, y_tp = combine_class_data(baseline_path, amusement_path, 
                                        stress_path, include_amusement=False)
        x_tr = np.concatenate([x_tr, x_tp])
        y_tr = np.concatenate([y_tr, y_tp])
    
    x_tr = x_tr[1:, ]
    y_tr = y_tr[1:, ]
    
    print("\nTrain data")
    print(x_tr.shape, y_tr.shape)
    
    # create the hot encoded labels and reshape the data
    x_tr = x_tr.reshape(-1, window_length, 1)
    x_ts = x_ts.reshape(-1, window_length, 1)
    y_tr_hot = gb_utl.get_hot_labels(y_tr)
    y_ts_hot = gb_utl.get_hot_labels(y_ts)
    
    # get the model
    model = get_eda_model()
    
    # train the model and evaluate it on the training and test data
    results = gb_utl.evaluate_model(model, x_tr, y_tr_hot, x_ts, y_ts_hot)
    
    # get the train and test report
    train_report = gb_utl.compute_performance_metrics(model, x_tr, y_tr)
    test_report = gb_utl.compute_performance_metrics(model, x_ts, y_ts)
    
    # store the results
    loso_result_dict[test_subject] = {'Tr Loss': results[0], "Tr Acc": results[1], "Ts Loss": results[2], "Ts Acc": results[3],
        "Tr TP": train_report[0], "Tr FP": train_report[1], "Tr TN": train_report[2], "Tr FN": train_report[3], 
        "Tr Recall": train_report[4], "Tr Precision": train_report[5], "Tr F1": train_report[6], "Tr ROC": train_report[7],
        "Tr Report": train_report[8],
        "Ts TP": test_report[0], "Ts FP": test_report[1], "Ts TN": test_report[2], "Ts FN": test_report[3], 
        "Ts Recall": test_report[4], "Ts Precision": test_report[5], "Ts F1": test_report[6], "Ts ROC": test_report[7],
        "Ts Report": test_report[8]}
    
print(loso_result_dict)
train_acc = []
test_acc = []

train_f1 = []
test_f1 = []

train_roc = []
test_roc = []

for subject in subject_ids:
    subject_data = loso_result_dict[subject]
    
    train_acc.append(subject_data['Tr Acc'] * 100)
    test_acc.append(subject_data['Ts Acc'] * 100)
    
    train_f1.append(subject_data['Tr F1'] * 100)
    test_f1.append(subject_data['Ts F1'] * 100)
    
    train_roc.append(subject_data['Tr ROC'] * 100)
    test_roc.append(subject_data['Ts ROC'] * 100)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axes[0].bar(subject_ids, train_acc, label="Train Acc")
axes[1].bar(subject_ids, test_acc, label="Test Acc")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axes[0].bar(subject_ids, train_f1, label="Train F1")
axes[1].bar(subject_ids, test_f1, label="Test F1")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axes[0].bar(subject_ids, train_roc, label="Train ROC")
axes[1].bar(subject_ids, test_roc, label="Test ROC")




