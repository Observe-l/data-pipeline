# Data-Pipeline
Kafka:https://kafka.apache.org/documentation/#gettingStarted

Kafka-Python:https://kafka-python.readthedocs.io/en/master/usage.html

MLFlow-Kafka:https://www.vantage-ai.com/en/blog/keeping-your-ml-model-in-shape-with-kafka-airflow-and-mlflow

Tensorflow-Kafka:https://www.tensorflow.org/io/tutorials/kafka

Zaurin's code: https://github.com/jrzaurin/ml-pipeline



# MLflow Tracking server

```shell
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /nfsroot/mlruns_sqlite \
    --host 0.0.0.0 \
    --port 6100
```

Use mysql database

```shell
mlflow server \
    --backend-store-uri mysql://VEC:666888@localhost:3306/MLflow \
    --default-artifact-root /nfsroot/mlruns \
    --host 0.0.0.0 \
    --port 5000
```

Mount nfs folder

```shell
sudo mount -t nfs 192.168.1.118:/nfsroot /nfsroot -o nolock,soft,timeo=30,retry=3
```

MLflow model serve

```shell
export MLFLOW_TRACKING_URI=http://192.168.1.118:5000

mlflow models serve -m "models:/mnist_best_model/Production" --port 6000
```

