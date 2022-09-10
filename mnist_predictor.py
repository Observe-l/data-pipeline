import json
import pandas as pd
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import mlflow
from pathlib import Path
from utils.kafka_producer import publish_messages, append_data
from kafka import KafkaConsumer, KafkaProducer


# The global parameters. Data path and server uri
KAFKA_HOST = 'redpc:9092'
TOPICS = ['mnist_app', 'mnist_train']
MODLE_NAME = 'mnist_best_model'
PATH = Path('mnist_data/')
APPEND_DATA = PATH/'messages'
TRAIN_DATA = PATH/'train'
RETRAIN_EVERY = 200

mlflow.set_tracking_uri("http://redpc:5000")
client = mlflow.tracking.MlflowClient()


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

def predict(model:mlflow.pyfunc.PyFuncModel,message):
    pred_data = np.array(message).reshape(1,28,28,1)
    pred = int(np.argmax(model.predict(pred_data)))
    return pred

if __name__ == '__main__':
    model = reload_model(model_name = MODLE_NAME)
    consumer = KafkaConsumer(bootstrap_servers=KAFKA_HOST)
    consumer.subscribe(TOPICS)
    message_count = 0
    batch_id = 0
    for kafka_msg in consumer:
        msg_value = json.loads(kafka_msg.value)
        if kafka_msg.topic == 'mnist_train' and 'training_completed' in msg_value and msg_value['training_completed']:
            latest_version = msg_value['model_version']
            model = reload_model(MODLE_NAME,latest_version)
            print(f'New model reloaded: version {latest_version}')
        elif kafka_msg.topic == 'mnist_app' and 'prediction' not in msg_value:
            request_id = msg_value['request_id']
            pred = predict(model,msg_value['data'])
            tmp_app = {'request_id': request_id, 'prediction': pred}
            print(f"The prediction is: {pred}, correct answer is: {msg_value['label']}")
            publish_messages(topic='mnist_app',messages=tmp_app)
            append_data(msg_value,TRAIN_DATA,APPEND_DATA,batch_id)
            message_count += 1
            if message_count % RETRAIN_EVERY == 0:
                tmp_train = {'retrain': True, 'batch_id': batch_id}
                publish_messages(topic='mnist_train', messages=tmp_train)
                batch_id += 1

