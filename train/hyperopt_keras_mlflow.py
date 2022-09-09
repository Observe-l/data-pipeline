import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import json
import pickle

from hyperopt import hp, tpe, fmin, Trials, SparkTrials, STATUS_OK

from pathlib import Path
from kafka import KafkaConsumer

from utils.kafka_producer import publish_messages

space = {
    "layer1_nodes": hp.quniform('layer1_nodes', 8,32,1), # return a integer value. round(uiform(low,up) / i ) * i
    "layer2_nodes": hp.quniform('layer2_nodes', 16,48,1),
    'dropout1': hp.uniform('dropout1', .01,.3)
}

# The global parameters. Data path and server uri
KAFKA_HOST = 'redpc:9092'
TOPICS = 'mnist_train'
MODLE_NAME = 'mnist_best_model'
PATH = Path('mnist_data/')
TRAIN_DATA = PATH/'train/mnist_data.p'
TRAIN_LABEL = PATH/'train/mnist_label.p'

def get_objective(data, label):
    def objective(params:dict):
        # mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("mnist_yellow")
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
            epochs = 5

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

