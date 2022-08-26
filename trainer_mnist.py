import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import mlflow

from hyperopt import hp, tpe, fmin, Trials, SparkTrials, STATUS_OK

mlflow.set_experiment("mnist-hyperopt")

space = {
    "l1_nodes": hp.uniform('l1_nodes', 8,16),
    'dropout1': hp.uniform('dropout1', .25,.75)
}
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnits.load_data()

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


def objective(params):
    mlflow.tensorflow.autolog()
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(params['l1_nodes'], kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
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

    score = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': -score[1], 'status': STATUS_OK, 'params': params}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials
)