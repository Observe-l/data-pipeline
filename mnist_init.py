import pickle
from pathlib import Path
import pandas as pd
import numpy as np

import mlflow
from tensorflow import keras
from tensorflow.keras import layers
from hyperopt import hp, tpe, fmin, Trials, SparkTrials, STATUS_OK

PATH = Path('mnist_data/')
APPEND_DATA = PATH/'messages'
TRAIN_DATA = PATH/'train'

space = {
    "layer1_nodes": hp.quniform('layer1_nodes', 8,32,1), # return a integer value. round(uiform(low,up) / i ) * i
    "layer2_nodes": hp.quniform('layer2_nodes', 16,48,1),
    'dropout1': hp.uniform('dropout1', .01,.3)
}

def get_objective(data, label):
    def objective(params:dict):
        # mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("mnist_kafka")
        num_classes = 10
        input_shape = (28, 28, 1)

        y_train = keras.utils.to_categorical(label, num_classes)
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.tensorflow.autolog(log_models=True, disable=False, registered_model_name=None)
            mlflow.log_params(params)
            model = keras.Sequential(
                [
                    keras.Input(shape=input_shape),
                    layers.Conv2D(params['layer1_nodes'], kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(params['layer2_nodes'], kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dropout(params['dropout1']),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )

            batch_size = 128
            epochs = 20

            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            history = model.fit(data, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
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
        f"mnist_best_model"
    )
    # Get the latest model version
    latest_version = int(result.version)
    # Updata the description
    client.update_model_version(
        name='mnist_best_model',
        version=latest_version,
        description=f"The hyperparameters: layer1 nodes:{best_result['params']['layer1_nodes']}, \
        layer2 nodes:{best_result['params']['layer2_nodes']}, \
        dropout1:{best_result['params']['dropout1']}"
    )
    # Transition the latest model to Production stage, others to Archived stage
    client.transition_model_version_stage(
        name='mnist_best_model',
        version= latest_version,
        stage='Production',
        archive_existing_versions=True
    )
    return latest_version

def create_folders():
    print("creating directory structure...")
    (PATH).mkdir(exist_ok=True)
    (APPEND_DATA).mkdir(exist_ok=True)
    (TRAIN_DATA).mkdir(exist_ok=True)

def download_data():
    print("downloading testing data...")
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Use first 10,000 data to train the model, 60,000 to retrain the model
    train_all = np.concatenate((x_train, x_test))
    train_all = np.expand_dims(train_all,-1)
    label_all = np.concatenate((y_train,y_test))
    train_save = np.split(train_all,[10000])
    train_label = np.split(label_all,[10000])
     
    # Save the training data and label
    pickle.dump(train_save[0],open(TRAIN_DATA/'mnist_data.p',"wb"))
    pickle.dump(train_label[0],open(TRAIN_DATA/'mnist_label.p',"wb"))
    # Save the transfer data and label
    pickle.dump(train_save[1],open(TRAIN_DATA/'kafka_data.p',"wb"))
    pickle.dump(train_label[1],open(TRAIN_DATA/'kafka_label.p',"wb"))

def create_model():
    x_train = pickle.load(open(TRAIN_DATA/'mnist_data.p',"rb"))
    y_train = pickle.load(open(TRAIN_DATA/'mnist_label.p',"rb"))
    model_version = hyper_opt(x_train=x_train, y_train=y_train, maxevals=10)
    print(f"Complete training, latest version is: {model_version}")


if __name__ == '__main__':
    mlflow.set_tracking_uri("http://redpc:5000")
    create_folders()
    download_data()
    create_model()
