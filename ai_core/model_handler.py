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

# class meta_data:
#     def __init__(self, id=None,label=None,service=None,algorithm=None,parameters=None,cols=None,time_range=None,desc=None):
#         self.id: str = id # model unique id
#         self.label: str = label # alias of id, unique in service
#         self.service: str = service # service name
#         self.algorithm: str = algorithm # learning algorithm name
#         self.parameters: dict = parameters # learning parameters
#         self.cols: list = cols # learning cols, if supervised laenring model cols[0] is target column name, e.g.) ['power generation', 'temperature', 'humidity']
#         self.time_range: list = time_range # learning data time range, e.g.) [ ['2022-03-15 05:14:13', '2022-06-10 06:38:25'], ['2023-01-14 00:00:00', '2023-02-14 06:00:00'] ]
#         self.desc: str = desc # description of model
    
#     def get_json(self):
#         return json.dumps(self.__dict__)
#     def get_dict(self):
#         return self.__dict__

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
    
    def set(self, algorithm=None, parameters=None, meta=None):
        temp_data_set = None
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
            raise RuntimeError(e)
    
    def remove_data(self):
        self.model.remove_data()

    def fit(self):
        self.model.fit()

    # TODO 필요없는듯?
    # def update(self):
    #     pass
    def predict(self, data, data_type, batch_size):
        try:
            res = []
            for _data in data:
                res.append( self.model.predict(data=_data, data_type=data_type, batch_size=batch_size) )
            return res
        except Exception as e:
            raise RuntimeError(e)