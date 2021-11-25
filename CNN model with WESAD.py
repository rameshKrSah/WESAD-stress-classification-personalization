
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import my_utils as gb_utl

data_folder = '../Processed Data/WESAD/Segments/60 seconds 50 % overlap/'
amusement_path = data_folder + "E4_amusement_eda_segments.pickle"
baseline_path = data_folder + "E4_baseline_eda_segments.pickle"
stress_path = data_folder + "E4_stress_eda_segments.pickle"

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
    

# create the X and Y values
X_gsr, Y_gsr = combine_class_data(baseline_path, amusement_path, stress_path, include_amusement=False)

df = pd.DataFrame(X_gsr)
df['Class'] = Y_gsr

n_classes = np.max(Y_gsr) + 1
window_length = X_gsr.shape[1]
LR_RATE = 0.001

X_gsr = X_gsr.reshape(-1, window_length, 1)
X_gsr.shape, Y_gsr.shape, np.unique(Y_gsr, return_counts=True)
x_gsr_train, x_gsr_val, x_gsr_test, y_gsr_train, y_gsr_val, y_gsr_test = gb_utl.split_into_train_val_test(X_gsr, Y_gsr)
x_gsr_train = x_gsr_train.reshape(-1, window_length, 1)
x_gsr_val = x_gsr_val.reshape(-1, window_length, 1)
x_gsr_test = x_gsr_test.reshape(-1, window_length, 1)
y_gsr_train_hot = gb_utl.get_hot_labels(y_gsr_train)
y_gsr_test_hot = gb_utl.get_hot_labels(y_gsr_test)
if len(y_gsr_val):
    y_gsr_val_hot = gb_utl.get_hot_labels(y_gsr_val)

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

def evaluate_model(model, x_tr, y_tr_hot, x_ts, y_ts_hot, BATCH_SIZE = 32, EPOCHS=50):
    # plot loss function
    loss = gb_utl.PlotLosses()
    cbs = [loss]
    
    # fit the model
    model_history = model.fit(x_tr, y_tr_hot, batch_size = BATCH_SIZE, epochs = EPOCHS, 
                                  verbose = 0, callbacks = cbs)
    
    # get the performance values
    tr_loss, tr_acc = model.evaluate(x_tr, y_tr_hot)
    ts_loss, ts_acc = model.evaluate(x_ts, y_ts_hot)
    
    return [tr_loss, tr_acc, ts_loss, ts_acc]

# Model training
model = get_eda_model()
print(model.summary())
metrics = evaluate_model(model, x_gsr_train, y_gsr_train_hot, 
                         x_gsr_test, y_gsr_test_hot, EPOCHS=200)

# Evalaute the model
train_report = gb_utl.compute_performance_metrics(model, x_gsr_train, y_gsr_train)
test_report = gb_utl.compute_performance_metrics(model, x_gsr_test, y_gsr_test)
cv_results = gb_utl.cross_validation(get_eda_model, X_gsr, Y_gsr, 3, 0.25, 0.0)

for i in range(3):
    print(cv_results[i]['Tr Report'])
    print(cv_results[i]['Ts Report'])
    
# save the trained model
# model.save("models/WESAD_EDA_CNN_60_50_MODEL")
