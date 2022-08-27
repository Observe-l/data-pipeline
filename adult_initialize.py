import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import warnings
import argparse
import os
import pdb
import mlflow
from tensorflow import keras

from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn.model_selection import train_test_split
from hyperopt import hp, tpe, fmin, Trials, SparkTrials, STATUS_OK

from pathlib import Path
from utils.preprocess_data import build_train


PATH = Path('data/')
TRAIN_PATH = PATH/'train'
DATAPROCESSORS_PATH = PATH/'dataprocessors'
MODELS_PATH = PATH/'models'
MESSAGES_PATH = PATH/'messages'

def data_sequence(model_id):
    """
    Data Providing function

    This function will be called by hyperas
    The train data is a 3D array. slide window size is 15
    train_test_spilit could generate test dataset and train dataset

    """
    init_dataprocessor = 'dataprocessor_{}_.p'.format(model_id)
    root_path = Path('data/')
    processor_path = root_path/'dataprocessors'

    trainDataset = pickle.load(open(processor_path/init_dataprocessor, 'rb'))
    train_X = trainDataset.data
    train_Y = trainDataset.target
    colnames = trainDataset.colnames
    data_array = np.array(train_X[colnames].values)
    label_array = np.array(train_Y.values).T

    data_train, data_test, label_train, label_test = train_test_split(data_array, label_array, test_size=0.05)
    return data_train, label_train, data_test, label_test

def hyperopt_model(data_train, label_train, data_test, label_test):
    """
    Create a keras model
    3 Dense layer
    hyperparameters:
    layer 1 nodes: 8,16,32
    layer 2 nodes: 16,36,64    
    Dropout layer probability: [0.1-0.5]
    batch size: 64, 128
    """
    mlflow.tensorflow.autolog()
    # cb = keras.callbacks.TensorBoard(log_dir= self.get_run_logdir("hyperopt_history"), histogram_freq=1, write_graph= True, update_freq='epoch')

    model = keras.models.Sequential()
    model.add(keras.layers.Dense({{choice([8,16,32])}},input_dim=15,activation='relu'))
    model.add(keras.layers.Dropout({{uniform(0.1,0.5)}}))
    model.add(keras.layers.Dense({{choice([16,32,64])}},activation='relu'))
    model.add(keras.layers.Dropout({{uniform(0.1,0.5)}}))
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    # model = keras.Sequential(
    #     keras.layers.Dense(choice[8,16,32],input_dim=15,activation='relu'),
    #     keras.layers.Dropout(uniform(0.1,0.5)),
    #     keras.layers.Dense(choice[16,32,64],activation='relu'),
    #     keras.layers.Dropout(uniform(0.1,0.5)),
    #     keras.layers.Dense(1,activation='sigmoid')
    # )
    model.compile(optimizer='adam',
                loss = 'binary_crossentropy',
                metrics=['accuracy'])
    
    history = model.fit(data_train, label_train,
                        batch_size = {{choice([64,128])}},
                        epochs = 2,
                        validation_data = (data_test,label_test)
    )
    validation_acc = np.amax(history.history['accuracy'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def create_folders():
    print("creating directory structure...")
    (PATH).mkdir(exist_ok=True)
    (TRAIN_PATH).mkdir(exist_ok=True)
    (MODELS_PATH).mkdir(exist_ok=True)
    (DATAPROCESSORS_PATH).mkdir(exist_ok=True)
    (MESSAGES_PATH).mkdir(exist_ok=True)


def download_data():
    train_path = PATH/'adult.data'
    test_path = PATH/'adult.test'

    COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
               "marital_status", "occupation", "relationship", "race", "gender",
               "capital_gain", "capital_loss", "hours_per_week", "native_country",
               "income_bracket"]

    print("downloading training data...")
    df_train = pd.read_csv("adult.data",
        names=COLUMNS, skipinitialspace=True, index_col=0)
    df_train.drop("education_num", axis=1, inplace=True)
    df_train.to_csv(train_path)
    df_train.to_csv(PATH/'train/train.csv')

    print("downloading testing data...")
    df_test = pd.read_csv("adult.test",
        names=COLUMNS, skipinitialspace=True, skiprows=1, index_col=0)
    df_test.drop("education_num", axis=1, inplace=True)
    df_test.to_csv(test_path)


def create_data_processor():
    print("creating preprocessor...")
    dataprocessor = build_train(TRAIN_PATH/'train.csv', DATAPROCESSORS_PATH)


def create_model(hyper):
    print("creating model...")
    init_dataprocessor = 'dataprocessor_0_.p'
    dtrain = pickle.load(open(DATAPROCESSORS_PATH/init_dataprocessor, 'rb'))
    if hyper == "hyperopt":
        # from train.train_hyperopt import LGBOptimizer
        from train.train_hyperopt_mlflow import LGBOptimizer
        LGBOpt = LGBOptimizer(dtrain, MODELS_PATH)
        LGBOpt.optimize(maxevals=50)
    elif hyper == "hyperparameterhunter":
        # from train.train_hyperparameterhunter import LGBOptimizer
        from train.train_hyperparameterhunter_mlfow import LGBOptimizer
        LGBOpt = LGBOptimizer(dtrain, MODELS_PATH)
        LGBOpt.optimize(maxevals=50)
    elif hyper == "keras":
        model_id = 0
        trials = Trials()
        mlflow.set_experiment("keras-init")
        best_run, best_model = optim.minimize(model=hyperopt_model,
                                              data= data_sequence,
                                              algo=tpe.suggest,
                                              max_evals=50,
                                              trials=trials,
                                              data_args=(model_id,))
        model_fname = 'model_{}_.p'.format(model_id)
        best_experiment_fname = 'best_experiment_{}_.p'.format(model_id)
        pickle.dump(best_model, open(MODELS_PATH/model_fname, 'wb'))
        pickle.dump(best_run, open(MODELS_PATH/best_experiment_fname, 'wb'))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--hyper", type=str, default="keras")
    args = parser.parse_args()
    create_folders()
    download_data()
    create_data_processor()
    create_model(args.hyper)
