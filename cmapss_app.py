import pandas as pd
import numpy as np
import uuid
import json
import time
import threading

from pathlib import Path
from .udp_req import udp_send

MODLE_NAME = 'mnist_best_model'
PATH = Path('CMAPSSData/')
APPEND_DATA = PATH/'messages'
TRAIN_DATA = PATH/'train_FD003.txt'

def start_sending():
    send_data = open(TRAIN_DATA,'rb')
    data_msg = send_data.readlines()
    for tmp_msg in data_msg:
        udp_send(tmp_msg, 'localhost')
        time.sleep(0.005)

if __name__ == '__main__':
    start_sending()