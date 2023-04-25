'''
handling model object
'''
from logger.logger import logger
import time
import copy
import pickle
import json
from os.path import exists

import torch
from torch import nn
import matplotlib.pyplot as plt
from ai_core.algorithm.Sequential_model import ai_model
from config import attribute

str_service = 'service'
str_label = 'label'
str_algorithm = 'algorithm'
str_data_path = 'data_path'
str_data_type = 'data_type'
str_parameters = 'parameters'


'''
algorithm_list = {
    "sequnecial" : sequential_rnn
    "RNN" : RNN
'''
algorithm_list = {
    'rnn' : ai_model,
    'gru' : ai_model,
    'lstm' : ai_model
}


class model_handler():
    def __init__(self, id, algorithm, parameters, meta, data=None):
        self.id: str = id
        self.meta :dict = {
            str_service : None,
            str_label : None,
            str_algorithm : None,
            str_data_path : None,
            str_data_type : None,
            str_parameters : None
        }
        self.algorithm: str
        self.parameters: dict
        self.model: ai_model

        self.set(algorithm, parameters, meta)
    
    def set(self, algorithm=None, parameters=None, meta=None, reset=False):
        temp_data_set = None
        if reset==False:
            try:
                temp_data_set = self.model.dataset
            except:
                pass

        if parameters!=None:
            if algorithm!=None:
                if algorithm not in algorithm_list.keys():
                    msg = f'given algorithm not available'
                    logger.error(msg)
                    raise ValueError(msg)
                self.algorithm = algorithm
            try:
                self.parameters = parameters
                self.model = algorithm_list[self.algorithm](self.algorithm, self.parameters)
                try:
                    self.model.dataset = temp_data_set
                except:
                    pass
            except Exception as e:
                msg = f'ai model initialization filed : {e}'
                logger.error(msg)
                raise ValueError(msg)

        if meta!=None:
            for _key in meta.keys():
                self.meta[_key] = meta[_key]

    def add_data(self, data_set:list, data_type:str):
        '''
        add data set to exist self.data_set for learning
        data shape must be same to exist data_set
        data_type = { 'csv' | 'pkl' | 'df' | 'list'}
        if data_type in ['csv', 'pkl'], data should be list of str (e.g. ['file1.pkl', 'file2.pkl'])
        '''
        try:
            if type(data_set) is list:
                for _data_set in data_set:
                    self.model.add_data(data=_data_set, data_type=data_type)
            else:
                self.model.add_data(data=data_set, data_type=data_type)
        except Exception as e:
            msg = f'model_handler.add_data failed : {e}'
            raise RuntimeError(msg)

    def remove_data(self):
        self.model.remove_data()

    def fit(self):
        try:
            self.model.fit()
        except Exception as e:
            msg = f'model_handler.fit failed : {e}'
            raise RuntimeError(msg)

    def predict(self, data, data_type, batch_size):
        try:
            if data_type in ['pkl', 'csv'] and type(data)==list:
                res = []
                for _data in data:
                    res.append( self.model.predict(data=_data, data_type=data_type, batch_size=batch_size) )
            else:
                res = self.model.predict(data=data, data_type=data_type, batch_size=batch_size)
            return res
        except Exception as e:
            raise RuntimeError(e)