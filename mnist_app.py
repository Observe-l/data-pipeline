import pickle
import json
import threading
import pandas as pd
import numpy as np
import uuid

from time import sleep
from pathlib import Path
from utils.kafka_producer import publish_messages, append_data
from kafka import KafkaConsumer, KafkaProducer

# The global parameters. Data path and server uri
KAFKA_HOST = 'redpc:9092'
TOPICS = ['mnist_app', 'mnist_train']
MODLE_NAME = 'mnist_best_model'
PATH = Path('mnist_data/')
DATAPROCESSORS_PATH = PATH/'dataprocessors'
APPEND_DATA = PATH/'messages'
TRAIN_DATA = PATH/'train'
KAFKA_DATA = PATH/'train/kafka_data.p'
KAFKA_LABEL = PATH/'train/kafka_label.p'



def start_producing():
    mnist_data = pickle.load(open(KAFKA_DATA,'rb'))
    mnist_data = mnist_data.reshape(mnist_data.shape[0],-1).tolist()
    mnist_label = pickle.load(open(KAFKA_LABEL,'rb')).tolist()


    for i in range(len(mnist_data)):
        message_id = str(uuid.uuid4())
        kafka_message = {'request_id': message_id, 'data': mnist_data[i], 'label': mnist_label[i]}
        publish_messages(topic='mnist_app',messages=kafka_message)
        print("\033[1;31;40m -- PRODUCER: Sent message with id {}".format(message_id))
        sleep(0.5)

def start_consuming():
    consumer = KafkaConsumer('mnist_app', bootstrap_servers=KAFKA_HOST)
    for msg in consumer:
        message = json.loads(msg.value)
        if 'prediction' in message:
            request_id = message['request_id']
            print("\033[1;32;40m ** CONSUMER: Received prediction {} for request id {}".format(message['prediction'], request_id))


if __name__ == '__main__':
    t1 = threading.Thread(target=start_producing)
    t2 = threading.Thread(target=start_consuming)
    t1.start()
    t2.start()

    # start_producing()