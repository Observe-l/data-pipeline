import json
import pandas as pd
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import mlflow
from pathlib import Path
import socket
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix

PATH = Path('CMAPSSData/')
TEST_DATA = PATH/'test_FD003.txt'
MODEL_PORT = 9768
DATA_PORT = 9860
TIME_OUT = 3
MODEL_NAME = "classification_test"
TEST_FILE = "CMAPSSData/test_FD003.txt"
RUL_FILE = "CMAPSSData/RUL_FD003.txt"
RESULT = "cl_test_result.txt"

def reload_model(model_name:str, model_version:str = None) -> mlflow.pyfunc.PyFuncModel:
    '''
    The models are stored in the MLflow tracking server.
    Fetch the latest model
    '''
    if model_version:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
    else:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/Production"
        )
    return model

def data_load(sequence_length=25):
    # Read data from txt file
    test_df = pd.read_csv(TEST_FILE, sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21']
    # read ground truth data
    truth_df = pd.read_csv(RUL_FILE, sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop('more', axis=1, inplace=True)
    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)
    # generate label1 column for test data
    test_df['label1'] = np.where(test_df['RUL'] <= 30, 1, 0 )
    scaler = preprocessing.MinMaxScaler()
    # MinMax normalization for test data
    test_df['cycle_norm'] = test_df['cycle']
    scaler_column = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
    for col in scaler_column:
        test_df[[col]] = scaler.fit_transform(test_df[[col]])
    # pick the feature columns 
    sensor_cols = ['s' + str(i) for i in range(1,22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)
    seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-sequence_length:] 
                        for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= sequence_length]
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
    # Last cycle test data - seq_array for test data
    print(seq_array_test_last.shape)
    y_mask = [len(test_df[test_df['id']==id]) >= sequence_length for id in test_df['id'].unique()]
    label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)
    return seq_array_test_last, label_array_test_last


def eval_model(model:mlflow.pyfunc.PyFuncModel, testdata, testlabel):
    label_pre = model.predict(testdata)
    # Set the threshold = 0.5
    label_pre = (label_pre>0.5).astype(np.int64)
    label_pre = label_pre[:,1]
    accuracy = np.sum(label_pre==testlabel) / label_pre.shape[0]
    pr_acc = "Accuracy is:" + str(accuracy) + "\n"
    print(pr_acc)
    result = confusion_matrix(testlabel, label_pre,normalize='pred')
    print("Confusion matrix is:\n",result,"\n")
    report = classification_report(testlabel, label_pre, target_names=target_name)
    pr_rep = "Classification report:\n" + report
    print(pr_rep)
    with open(RESULT,'a') as f:
        f.write(pr_acc)
        f.write(pr_rep)


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:5000")
    f = open(RESULT,'w')
    f.close()
    model = reload_model(model_name = MODEL_NAME)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost",MODEL_PORT))
    sock.listen(1)
    test_data, test_label = data_load()
    target_name = ['Normal','Broken']
    eval_model(model,test_data,test_label)

    c, addr = sock.accept()
    while True:
        byte_read = c.recv(4096)
        if not byte_read:
            break
        else:
            print(byte_read.decode())
            model = reload_model(model_name = MODEL_NAME)
            eval_model(model,test_data,test_label)
            