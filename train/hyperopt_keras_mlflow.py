import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import time
import pickle
import pdb
import warnings
import mlflow
import mlflow.sklearn

from tensorflow import keras

from hyperas import optim
from hyperas.distributions import choice, uniform

from sklearn.model_selection import train_test_split

from pathlib import Path
from sklearn.metrics import f1_score
from hyperopt import hp, tpe, fmin, Trials, SparkTrials, STATUS_OK
from mlflow.tracking import MlflowClient


warnings.filterwarnings("ignore")

class LGBOptimizer(object):
	def __init__(self, trainDataset, out_dir):
		"""
		Hyper Parameter optimization with keras

		Parameters:
		-----------
		trainDataset: FeatureTools object
			The result of running FeatureTools().fit()
		out_dir: pathlib.PosixPath
			Path to the output directory
		"""
		self.PATH = out_dir
		self.early_stop_dict = {}
		self.seq_len = 15

		self.X = trainDataset.data
		self.y = trainDataset.target
		self.colnames = trainDataset.colnames
		self.categorical_columns = trainDataset.categorical_columns + trainDataset.crossed_columns

		self.train_array = self.gen_sequence(self.X, self.seq_len, self.colnames)
		self.label = np.delete(np.array(self.y.values).T,np.s_[0:self.seq_len-1],0)

		self.lgtrain = lgb.Dataset(self.X,label=self.y,
			feature_name=self.colnames,
			categorical_feature = self.categorical_columns,
			free_raw_data=False)

	def gen_sequence(self,df_train: pd.DataFrame ,seq_length: int, seq_cols):
		
		data_array = df_train[seq_cols].values
		num_element = data_array.shape[0]
		train_array = np.zeros((num_element-seq_length+1,seq_length,len(seq_cols)))
		for idx in range(num_element-seq_length+1):
			train_array[idx,:,:] = data_array[idx:idx+seq_length,:]
		return train_array
	
	def data_sequence(self):
		"""
		Data Providing function

		This function will be called by hyperas
		The train data is a 3D array. slide window size is 15
		train_test_spilit could generate test dataset and train dataset

		"""
		data_array = np.array(self.X[self.colnames].values)
		label_array = np.array(self.y.values).T

		data_train, data_test, label_train, label_test = train_test_split(data_array, label_array, test_size=0.05)
		return data_train, label_train, data_test, label_test

	def get_run_logdir(self, k):
		"""
		Record the data in the tensorboard
		"""
		root_logdir = os.path.join(os.curdir, "hyperopt_keras", k)
		run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
		return os.path.join(root_logdir, run_id)

	def create_model(self, data_train, label_train, data_test, label_test):
		"""
		Create a keras model
		3 Dense layer
		hyperparameters:
		layer 1 nodes: 8,16,32
		layer 2 nodes: 16,36,64
		Dropout layer probability: [0.1-0.5]
		batch size: 64, 128
		"""
		cb = keras.callbacks.TensorBoard(log_dir= self.get_run_logdir("hyperopt_history"), histogram_freq=1, write_graph= True, update_freq='epoch')

		model = keras.Sequential(
			keras.layers.Dense(choice[8,16,32],input_dim=15,activation='relu'),
			keras.layers.Dropout(0.1),
			keras.layers.Dense(choice[16,32,64],activation='relu'),
			keras.layers.Dropout(uniform(0.1,0.5)),
			keras.layers.Dense(1,activation='sigmoid')
		)
		model.compile(optimizer='adam',
					loss = 'binary_crossentropy',
					metrics=['accuracy'])
		
		history = model.fit(data_train, label_train,
							batch_size = {choice([64,128])},
							epochs = 2,
							callbacks = [cb],

		)
		validation_acc = np.amax(history.history['val_accuracy'])
		print('Best validation acc of epoch:', validation_acc)
		return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


	def optimize(self, maxevals=200, model_id=0, reuse_experiment=False):
		trials = Trials()
		# trials = SparkTrials(parallelism=2)
		best_run, best_model = optim.minimize(
			model=self.create_model,
			data=self.data_sequence,
			algo=tpe.suggest,
			max_evals=maxevals,
			trials= trials)
		model_fname = 'model_{}_.p'.format(model_id)
		best_experiment_fname = 'best_experiment_{}_.p'.format(model_id)
		pickle.dump(best_model, open(self.PATH/model_fname, 'wb'))
		pickle.dump(best_run, open(self.PATH/best_experiment_fname, 'wb'))



	def hyperparameter_space(self, param_space=None):

		space = {
			'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
			'boost_round': hp.quniform('boost_round', 50, 500, 20),
			'num_leaves': hp.quniform('num_leaves', 31, 256, 4),
		    'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
		    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
		    'subsample': hp.uniform('subsample', 0.5, 1.),
		    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
		    'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
		}

		if param_space:
			return param_space
		else:
			return space
