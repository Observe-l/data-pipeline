import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import json
import pickle

from hyperopt import hp, tpe, fmin, Trials, SparkTrials, STATUS_OK

from pathlib import Path

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
        num_classes = 10
        i_shape = [25, 25]

        y_train = keras.utils.to_categorical(label, num_classes)
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

            history = model.fit(data, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.05)
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
        description=f"The hyperparameters: filter1:{best_result['params']['filter1']}, \
        filter2:{best_result['params']['filter2']}, \
        filter3:{best_result['params']['filter3']}, \
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


