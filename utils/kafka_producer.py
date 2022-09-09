import json
import pickle
import pandas as pd
import numpy as np
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='redpc:9092')

def publish_messages(topic: str, messages: dict):
    producer.send(topic,json.dumps(messages).encode('utf-8'))
    producer.flush()

def append_data(message, path, archive_path, batch_id:int = 0):
    # Load original data
    data_path = path/'mnist_data.p'
    label_path = path/'mnist_label.p'
    train_data = pickle.load(open(data_path,'rb'))
    train_label = pickle.load(open(label_path,'rb'))

    # Load new message
    new_data = np.array(message['data']).reshape(1,28,28,1)
    new_label = np.array(message['label']).reshape(1)
    # Append new data
    train_data = np.concatenate((train_data,new_data))
    train_label = np.concatenate((train_label,new_label))
    pickle.dump(train_data,open(data_path,'wb'))
    pickle.dump(train_label,open(label_path,'wb'))

    # Save new messages
    data_path = archive_path/'message_{}_data.p'.format(batch_id)
    label_path = archive_path/'message_{}_label.p'.format(batch_id)
    pickle.dump(new_data,open(data_path,'wb'))
    pickle.dump(new_label,open(label_path,'wb'))



