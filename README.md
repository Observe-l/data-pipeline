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
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

