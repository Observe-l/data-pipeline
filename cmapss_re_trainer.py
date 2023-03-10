import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
import mlflow
import socket
import json
import pickle

from utils.udp_req import udp_send, udp_server

from hyperopt import hp, tpe, fmin, Trials, SparkTrials, STATUS_OK

from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

MODEL_PORT = 10652
DATA_PORT = 10653

FILE_NAME = "re_train.txt"
DATA_TMP = "re_rec.txt"
ORIGIN_DATA = 'CMAPSSData/train_FD003.txt'
TIME_OUT = 3
MODEL_NAME = "regression_loss_0"
REMAIN_NUM = 35

space = {
    "filter1": hp.quniform('filter1', 8,32,1), # return a integer value. round(uiform(low,up) / i ) * i
    "filter2": hp.quniform('filter2', 16,64,1),
    "filter3": hp.quniform('filter3', 32,128,1),
    "dropout1": hp.uniform('dropout1', .01,.5),
    "dropout2": hp.uniform('dropout2', .01,.5),
    "dropout3": hp.uniform('dropout3', .01,.5)
}

def get_objective(data, label):
    def objective(params:dict):
        # mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("cmapss_udp")
        i_shape = [25, 25]

        # y_train = keras.utils.to_categorical(label, num_classes)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.tensorflow.autolog(log_models=True, disable=False, registered_model_name=None)
            mlflow.log_params(params)
            model = keras.Sequential(
                [
                    layers.Conv1D(params['filter1'],5, activation='relu',padding='causal',input_shape=i_shape),
                    layers.Dropout(params['dropout1']),
                    layers.Conv1D(params['filter2'],7, activation='relu',padding='causal'),
                    layers.Dropout(params['dropout2']),
                    layers.Conv1D(params['filter3'],11, activation='relu',padding='causal'),
                    layers.Dropout(params['dropout3']),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(1,activation='sigmoid')
                ]
            )

            batch_size = 128
            epochs = 20

            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

            history = model.fit(data, label, batch_size=batch_size, epochs=epochs, validation_split=0.05)
            # mlflow.sklearn.log_model(model,"model")
            # score = model.evaluate(x_test, y_test, verbose=0)
            score = -history.history['accuracy'][-1]
        objective.i += 1
        return {'loss': score, 'status': STATUS_OK, 'params': params, 'mlflow_id': run_id}
    return objective


def hyper_opt(x_train,y_train, maxevals:int = 5):
    client = mlflow.tracking.MlflowClient()

    trials = Trials()
    objective = get_objective(x_train,y_train)
    objective.i=0
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=maxevals,
        trials=trials
    )
    # Get the best parameters and model id
    best_result = trials.best_trial['result']
    # Register the best model
    result = mlflow.register_model(
        f"runs:/{best_result['mlflow_id']}/model",
        f"CMAPSS_best_model"
    )
    # Get the latest model version
    latest_version = int(result.version)
    # Updata the description
    client.update_model_version(
        name='CMAPSS_best_model',
        version=latest_version,
        description=f"The hyperparameters: filter1:{best_result['params']['filter1']}, \
        filter2:{best_result['params']['filter2']}, \
        filter3:{best_result['params']['filter3']}, \
        dropout1:{best_result['params']['dropout1']}, \
        dropout2:{best_result['params']['dropout2']}, \
        dropout3:{best_result['params']['dropout3']}"
    )
    # Transition the latest model to Production stage, others to Archived stage
    client.transition_model_version_stage(
        name='CMAPSS_best_model',
        version= latest_version,
        stage='Production',
        archive_existing_versions=True
    )
    return latest_version


def single_train(x_train,y_train):
    client = mlflow.tracking.MlflowClient()
    mlflow.set_experiment("cmapss_udp")

    i_shape = [25, 25]

    # y_train = keras.utils.to_categorical(label, num_classes)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.tensorflow.autolog(log_models=True, disable=False, registered_model_name=None)
        model = keras.Sequential(
            [
                layers.LSTM(4,return_sequences=True,activation='relu',input_shape=i_shape),
                layers.Dropout(0.2),
                layers.LSTM(8,return_sequences=True,activation='relu'),
                layers.Dropout(0.2),
                layers.LSTM(16,return_sequences=False,activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(8, activation="relu"),
                layers.Dense(1,activation='linear')
            ]
        )

        batch_size = 128
        epochs = 20

        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.05)
        # mlflow.sklearn.log_model(model,"model")
        # score = model.evaluate(x_test, y_test, verbose=0)
        # score = -history.history['accuracy'][-1]
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        MODEL_NAME
    )
    # Get the latest model version
    latest_version = int(result.version)
    client.update_model_version(
        name=MODEL_NAME,
        version=latest_version,
        description='None'

    )
    # Transition the latest model to Production stage, others to Archived stage
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version= latest_version,
        stage='Production',
        archive_existing_versions=True
    )
    return latest_version


# function to reshape features into seq_array: (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means
    we need to drop those which are below the window-length. """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# function to generate label_array
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

def data_load(file_name:str = FILE_NAME, sequence_length=25):
    # Read data from txt file
    train_df = pd.read_csv(file_name, sep=" ",header=None)
    # print('latest machine: ',np.sort(train_df[0].unique())[::1][0])
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21']
    # Generate labels. If RUL < trashold, label = 1, o.w 0
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')

    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)

    # generate label1 column for training data
    train_df['label1'] = np.where(train_df['RUL'] <= 30, 1, 0 )
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    scaler_column = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21']
    for col in scaler_column:
        train_df[[col]] = scaler.fit_transform(train_df[[col]])
    
    # pick the feature columns 
    sensor_cols = ['s' + str(i) for i in range(1,22)]
    sequence_cols = ['setting1', 'setting2', 'setting3','cycle']
    sequence_cols.extend(sensor_cols)
    # generator for the training sequences
    seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)) 
            for id in train_df['id'].unique())
    # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    # generate labels (generated from "label1" col as its binary classification)
    label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['RUL']) 
                for id in train_df['id'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return seq_array, label_array

def update_data(train_data:pd.DataFrame):
    # get latest machine id
    tmp_idx = np.sort(train_data[0].unique())[::-1][:REMAIN_NUM]
    # update file
    train_data.loc[train_data[0].isin(tmp_idx)].to_csv(FILE_NAME, sep=' ', header=False,index=False)

if __name__ == "__main__":
    # Tracking the mysql database
    mlflow.set_tracking_uri("http://localhost:5000")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    start_flag = True
    # initialize data file
    f = open(DATA_TMP,'w')
    f.close()
    # Init data file
    tmp_data = pd.read_csv(ORIGIN_DATA, sep=" ",header=None)
    tmp_idx = np.sort(tmp_data[0].unique())[:REMAIN_NUM]
    tmp_data.loc[tmp_data[0].isin(tmp_idx)].to_csv(FILE_NAME, sep=' ', header=False,index=False)
    tmp_data = pd.read_csv(FILE_NAME, sep=" ",header=None)
    # Init model
    x_train, y_train = data_load()
    model_version = single_train(x_train, y_train)
    sock.connect(("localhost",MODEL_PORT))


    while(True):
        data, addr = udp_server(port=DATA_PORT,timeout=TIME_OUT,start_flag=start_flag)
        if data.decode('utf-8') == 'complete':
            tmp_buf = pd.read_csv(DATA_TMP, sep=" ",header=None)
            tmp_data = pd.concat([tmp_data, tmp_buf])
            # initialize data file
            f = open(DATA_TMP,'w')
            f.close()
            # update train data
            update_data(tmp_data)
            start_flag = True
            x_train, y_train = data_load()
            print(x_train.shape)
            # model_version = hyper_opt(x_train=x_train,y_train=y_train,maxevals=2)
            model_version = single_train(x_train, y_train)
            tmp_train_msg = f'model_version: {model_version}'
            sock.sendall(tmp_train_msg.encode())
            print(f"Retraining completed. The latest model version is: {model_version}")
        else:
            start_flag = False
            with open(DATA_TMP,'a') as f:
                f.write(data.decode('utf-8'))
