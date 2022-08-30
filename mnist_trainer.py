import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import mlflow

from hyperopt import hp, tpe, fmin, Trials, SparkTrials, STATUS_OK

space = {
    "layer1_nodes": hp.quniform('layer1_nodes', 8,32,1), # return a integer value. round(uiform(low,up) / i ) * i
    "layer2_nodes": hp.quniform('layer2_nodes', 16,48,1),
    'dropout1': hp.uniform('dropout1', .01,.3)
}
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def objective(params:dict):
    mlflow.set_experiment("log_models")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.tensorflow.autolog(log_models=False, disable=False, registered_model_name=None)
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

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        mlflow.sklearn.log_model(model,"model")
        score = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': -score[1], 'status': STATUS_OK, 'params': params, 'mlflow_id': run_id}

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=2,
        trials=trials
    )
    result = mlflow.register_model(
        f"runs:/{trials.best_trial['result']['mlflow_id']}/model",
        f"mnist_best_model"
    )