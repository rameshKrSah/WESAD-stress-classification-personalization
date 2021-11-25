import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras


class PlotLosses(keras.callbacks.Callback):
    """
        Keras Callback to plot the training loss and accuracy for the training and validation sets.
    """

    def __init__(self):
        self.i = 0
        self.epoch = []
        self.losses = []
        self.val_losses = []
        self.accu = []
        self.val_accu = []
        self.fig = plt.figure()
        self.logs = []
        self.tf_version = float(tf.__version__[:3])

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.epoch.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        if self.tf_version < 2.0:
            self.accu.append(logs.get('acc'))
            self.val_accu.append(logs.get('val_acc'))
        else:
            self.accu.append(logs.get('accuracy'))
            self.val_accu.append(logs.get('val_accuracy'))
        self.i += 1
        
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
        clear_output(wait=True)
        
        ax1.plot(self.epoch, self.accu, label="Acc")
        ax1.plot(self.epoch, self.val_accu, label="Val Acc")
        ax1.legend()
        
        ax2.plot(self.epoch, self.losses, label="Loss")
        ax2.plot(self.epoch, self.val_losses, label="Val Loss")
        ax2.legend()
        ax2.set_xlabel("Epoch")
        plt.show()
		
		
		
def print_confusion_matrix(confusion_matrix, class_names, title, activities, figsize = (12, 6), fontsize=10):
    """
    Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the output figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    
    fig = fig = plt.gcf()
    heatmap.yaxis.set_ticklabels(activities, rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(activities, rotation=90, ha='right', fontsize=fontsize)
    plt.show()
	
	
def get_features_labels_from_df(data_df, n_channels, n_window_len):
    """
	    Given a dataframe with class as column, separate the features and class label
	    and normalize the feature with min-max scaler and encode label as one-hot 
	    vector.

        Arguments:
        data_df (pandas DataFrame): dataframe
        n_channels (int) : Number of channels for the sensor data
        n_window_len (int) : Length of the window segment

        Returns:
        Normalized features in the range (-1.0, 1.0), label, and one hot encoded label
    """
    labels = data_df['Class'].values.astype(int)
    features = data_df.drop(['Class'], axis = 1).values
    
    scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    features = scaler.fit_transform(features)
    
    features = features.reshape(-1, n_channels, n_window_len)
    features = np.transpose(features, (0, 2, 1))
	
    labels_one_hot = keras.utils.to_categorical(labels, np.max(labels)+1)
    
    return features, labels, labels_one_hot	
	
	
def get_cnn_model(input_shape, n_output_classes, learning_rate):
    """ 
        Returns a 1D CNN model with arch 100 - 50 - GlobalMaxPool1D - 64 - Dropout(0.3) - n_classes. 
        We have used this 1D CNN model extensively in Adversarial research projects.

        Arguments: 
        input_shape (tuple) : Shape of the input
        n_output_classes (int) : number of output classes 
        learning_rate (float) : learning rate for the Adam optimizer

        Returns: 
        A 1D CNN model ready for training, with categorical cross entropy loss and Adam optimizer.
    """
    temp_model = keras.Sequential([
        keras.layers.Conv1D(filters = 100, kernel_size = (10), strides = 2, activation = tf.nn.relu, input_shape = input_shape),
        keras.layers.Conv1D(filters = 50, kernel_size = (5), strides = 1, activation = tf.nn.relu),
        keras.layers.GlobalMaxPool1D(),
        #keras.layers.Flatten(),
        keras.layers.Dense(units = 64, activation = tf.nn.relu),
        keras.layers.Dropout(rate = 0.3),
        keras.layers.Dense(units = n_output_classes, activation = tf.nn.softmax)
    ])
    
    temp_model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adam(learning_rate=learning_rate), 
                      metrics = ['accuracy'])
    
    return temp_model


def save_data(path, data):
    """
        Given a path and data, save the data to the path as a pickle file.

        Arguments:
        path (string) : file path with .pkl extension
        data : data values; can be a single container or multiple containers
    """
    f = open(path, "wb")
    pickle.dump(data, f)
    f.close()

def read_data(path, n_vaues=None):
    """
        Given a path, read the file and return the contents.

        Arguments:
        path (string) : File path with .pkl extension
        n_values (int) : Number of containers expected to be read. 
    """
    
    f = open(path, "rb")
    d = pickle.load(f)
    f.close()

    return d

def stylize_axis(ax, xticks=True, yticks=False, top_right_spines=True,
                    bottom_left_spines=False):
    """
        Given an axis, stylize it by removing ticks and spines. Default choice for
        ticks and spines are given. Modify as needed.

        Arguments:
        ax (matplotlib.axes.Ax): matplotlib axis
        xticks (Boolean): whether to make xticks visible or not (True by Default)
        yticks (Boolean): whether to make yticks visible or not (False by Default)
        top_right_spines (Boolean): whether to make top_right_spines visible or not (True by Default)
        bottom_left_spines (Boolean): whether to make bottom_left_spines visible or not (False by Default)

    """
    if xticks:
        ax.set_xticks([])

    if yticks:
        ax.set_yticks([])

    if top_right_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if bottom_left_spines:
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)



def print_metrics(metrics):
    """
    	@brief: Given a metrics, print the values. The metrics is the array obtained from 
    	the compute_performance_metrics function.
    """
    print("Training set loss: {:.3f}".format(metrics[0]))
    print("Training set accuracy: {:.3f} %".format(metrics[1] * 100))

    print("Test set loss: {:.3f}".format(metrics[2]))
    print("Test set accuracy: {:.3f} %".format(metrics[3] * 100))

    print("Precision score: {:.3f}".format(metrics[4]))
    print("Recall score: {:.3f}".format(metrics[5]))
    print("F1 score: {:.3f}".format(metrics[6]))
    print("ROC AUC: {:.3f}".format(metrics[7]))



def compute_performance_metrics(model, x, y):
    """
        Given a model (TensorFlow) and (x, y), we compute accuracy, loss, True Positive, False Negative,
        False Positive, True Negative, Recall, Precision, f1 score, Average Precision Recall, ROC AUC, 
        and classification report.

        Arguments:
            model: tensorflow model
            x: feature vector
            y: label vector

        Returns:
            True Positive, False Positive, False Negative, True Negative, Recall, Precision, f1 score, roc_auc_score
    """
    try:
        loss, acc = model.evaluate(x, y)
    except:
        y_hot = keras.utils.to_categorical(y, np.max(y) + 1)
        loss, acc = model.evaluate(x, y_hot)

    print("Accuracy {:.3f}, Loss {:.3f}".format(acc, loss))

    y_probs = model.predict(x)
    y_pred = np.argmax(y_probs, axis = 1)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    print("True Positive ", tp)
    print("False Positive ", fp)
    print("True Negative ", tn)
    print("False Negative ", fn)

    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)

    print("Recall {:.3f}, with formula {:.3f}".format(recall, (tp / (tp + fn))))
    print("Precision {:.3f}, with formula {:.3f}".format(precision, (tp / (tp + fp))))

    f1_score_cal = f1_score(y, y_pred)
    print("F1 socre {:.3f}, with formula {:.3f}".format(f1_score_cal,
                                                    2 * ((precision * recall) / (precision + recall))))

    print("Average precision score {:.3f}".format(average_precision_score(y, y_pred)))

    roc_auc = roc_auc_score(y, y_pred)
    print("ROC AUC Score {:.3f}".format(roc_auc))
    
    clf_report = classification_report(y, y_pred)
    print("Classification report \n", clf_report)

    return [tp, fp, tn, fn, recall, precision, f1_score_cal, roc_auc, clf_report]


def split_into_train_val_test(X, Y, test_split = 0.25, val_split=0.0):
    """ 
        Given data (X, Y), split the data into training, validation and test sets.
        Validation is 10 percent of the training set.

        Arguments:
            X (numpy.ndarray): Data vector
            Y (numpy.ndarray): Label vector
            test_split (float): Test split (0.25 by default)
            val_split (float): Validation set split (0.0 by default)

        Returns:
            x_train, y_train, x_val, y_val, x_test, and y_test
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must be the same length")

    # # Scale the data into range [-1.0, 1.0]
    # p, q, r = X.shape
    # X = X.reshape(-1, q * r)
    # scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    # X = scaler.fit_transform(X)
    # X = X.reshape(-1, q, r)

    # # transpose the X into (:, n_window_size, n_channels) to make it compatible with CNN model
    # if len(X.shape) == 3:
    #     X = np.transpose(X, (0, 2, 1))

    # split the data
    random_state = 42
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_split, random_state=random_state, shuffle=True, stratify=Y)
    
    x_val = np.array([])
    y_val = np.array([])
    if val_split > 0.0:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=random_state, shuffle=True, stratify=y_train)

    print("Training set {} \nTest set {}\nValidation set {}".format(x_train.shape, x_test.shape, x_val.shape))
    return x_train, x_val, x_test, y_train, y_val, y_test

def select_random_samples(data, n_samples):
    """
        @brief: Select n_samples random samples from the data
        @param: data (array)
        @param: n_samples (int) Number of samples to randomly select from the data.

        @return: Randomly selected samples
    """
    length = len(data)
    print(length, n_samples)
    random_index = np.random.randint(0, length, size=length if length < n_samples else n_samples)
    return data[random_index]

def get_hot_labels(Y):
    """
        Given label vector, return the one hot encoded label vector.

        Arguments:
            Y (numpy.ndarray): label vector
        
        Returns:
            One hot encoded label vector.
    """
    return keras.utils.to_categorical(Y, np.max(Y) + 1, dtype=int)

def find_min_max(X):
    """ Return the minimum and maximum value of X """
    return np.min(X), np.max(X)

def load_data_with_preprocessing(data_path):
    """
        Given a data path, load the data (must be in the format (X, Y)) and 
        scale the X in range [-1.0, 1.0] and return scaled x and y.

        Arguments:
            data_path (string): Pickle file path
    
        Returns:
            (X, Y)
    """
    # load the file
    f = open(data_path, "rb")
    try:
        x, y = pickle.load(f)
        f.close()
    except:
        f.close()
        return

    # check for same length
    if len(x) != len(y):
        raise ValueError("Unequal X and Y sizes")
    
#     print(x.shape, y.shape)
#     wherenane = np.argwhere(np.isnan(x))[:, 1]
#     print(np.unique(wherenane, return_counts=True))
    # do we need preprocessing 
    print("Before Scaling: Min - Max {}".format(find_min_max(x)))
    scaler = MinMaxScaler((-1.0, 1.0))
    x = scaler.fit_transform(x)
    print("After Scaling: Min - Max {}".format(find_min_max(x)))
    
    return x, y


def cross_validation(model_function, X, Y, n_CV, test_split, val_split, batch_size = 32, epochs = 50):
    """
        @brief: Do cross validation for n_CV times and returns the results.

        @param: model_function : A function that returns the model after calling it.
        @param: X (array): Total data
        @param: Y (array): Total label
        @param: test_split (float): The percentage of samples to be included in the test set
        @param: val_split (float): The percentage of samples to be included in the validation set.
        @param: batch_size (int): Default 32
        @param: epochs (int): Default 50

        @return: Results of the cross validation, a dictionary
    """
    x_tr, x_val, x_ts, y_tr, y_val, y_ts = split_into_train_val_test(X, Y, test_split, val_split)

    y_tr_hot = get_hot_labels(y_tr)
    y_val_hot = []
    if val_split != 0.0:
        y_val_hot = get_hot_labels(y_val)
    y_ts_hot = get_hot_labels(y_ts)

    results_dict = {}
    metrics_arr = []
    for i in range(n_CV):
        model = model_function()
        results = evaluate_model(model, x_tr, y_tr_hot, x_ts, y_ts_hot, x_val, y_val_hot, batch_size, epochs)
        metrics_arr.append(results)
        train_report = compute_performance_metrics(model, x_tr, y_tr)
        test_report = compute_performance_metrics(model, x_ts, y_ts)
        results_dict[i] = {'Tr Loss': results[0], "Tr Acc": results[1], "Ts Loss": results[2], "Ts Acc": results[3],
        "Tr TP": train_report[0], "Tr FP": train_report[1], "Tr TN": train_report[2], "Tr FN": train_report[3], 
        "Tr Recall": train_report[4], "Tr Precision": train_report[5], "Tr F1": train_report[6], "Tr ROC": train_report[7],
        "Tr Report": train_report[8],
        "Ts TP": test_report[0], "Ts FP": test_report[1], "Ts TN": test_report[2], "Ts FN": test_report[3], 
        "Ts Recall": test_report[4], "Ts Precision": test_report[5], "Ts F1": test_report[6], "Ts ROC": test_report[7],
        "Ts Report": test_report[8]}

    metrics_arr = np.array(metrics_arr).reshape(n_CV, 4)
    print("Average Training Set Accuracy {:.3f}".format(np.average(metrics_arr[:, 1].ravel())))
    print("Average Testing Set Accuracy {:.3f}".format(np.average(metrics_arr[:, 3].ravel())))

    return results_dict

def evaluate_model(model, x_tr, y_tr_hot, x_ts, y_ts_hot, x_val = [], y_val_hot = [], BATCH_SIZE = 32, EPOCHS = 50):
    """
        @brief: Train the model and evaluate it on training and test set and return the results.

        @param: model: TF model
        @param: x_tr: training x
        @param: y_tr_hot: Hot encoded training y
        @param: x_ts: test x
        @param: y_ts_hot: Hot encoded test y
        @param: x_val: validation x
        @param: y_val_hot: Hot encoded validation y
        @param: BATCH_SIZE (int): default value 32
        @param: EPOCHS (int): default value 50

        @return: [tr_loss, tr_acc, ts_loss, ts_acc] : Training loss, Training Acc, Test loss, Test Acc
    """
    # plot loss function
    loss = PlotLosses()
    cbs = [loss]
    
    if len(x_val) > 0:
          # fit the model
        model_history = model.fit(x_tr, y_tr_hot, batch_size = BATCH_SIZE, epochs = EPOCHS, 
                                  validation_data = (x_val, y_val_hot), verbose = 0, callbacks = cbs)
    else:
        # fit the model
        model_history = model.fit(x_tr, y_tr_hot, batch_size = BATCH_SIZE, epochs = EPOCHS, 
                                    verbose = 0, callbacks = cbs)
    
    # get the performance values
    tr_loss, tr_acc = model.evaluate(x_tr, y_tr_hot)
    ts_loss, ts_acc = model.evaluate(x_ts, y_ts_hot)
    
    return [tr_loss, tr_acc, ts_loss, ts_acc]

def segment_sensor_reading(values, window_duration, overlap_percentage,
                           sampling_frequency):
    """
        Sliding window segmentation of the values array for the given window
        duration and overlap percentage.

    param values: 1D array of values to be segmented
    param window_duration: Window duration in seconds
    param overlap_percentage: Float value in the range (0 < overlap_percentage < 1)
    param sampling_frequency: Frequency in Hz
    """
    total_length = len(values)
    window_length = sampling_frequency * window_duration
    segments = []
    if(total_length < window_length):
        return segments
    
    start_index = 0
    end_index = start_index + window_length
    increment_size = int(window_length * (1-overlap_percentage))
    
    while(1):
        # print(start_index, end_index)
	
        # get the segment
        v = values[start_index:end_index]

        # save the segment
        segments.append(v)

        # change the start and end index values
        start_index += increment_size
        end_index += increment_size 

        if (start_index > total_length) | (end_index > total_length):
        #print("we are done, no more segments possible")
            break
        
    segments = np.array(segments).reshape(len(segments), window_length)
    # print("# segments ", segments.shape[0])
    return segments



if __name__ == "__main__":
    print("Script with utilities functions used throughout the research projects.")
    print("Availabel Functions are:")
    print(get_cnn_model.__doc__)
    print(get_features_labels_from_df.__doc__)
    print(print_confusion_matrix.__doc__)
    print(PlotLosses.__doc__)
    print(save_data.__doc__)
    print(read_data.__doc__)
    print(stylize_axis.__doc__)
    print(print_metrics.__doc__)
    print(compute_performance_metrics.__doc__)
    print(split_into_train_val_test.__doc__)
    print(get_hot_labels.__doc__)
    print(find_min_max.__doc__)
    print(load_data_with_preprocessing.__doc__)
    print(evaluate_model.__doc__)
    print(cross_validation.__doc__)
    print(segment_sensor_reading.__doc__)
    
