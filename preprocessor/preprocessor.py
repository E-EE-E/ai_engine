from typing import Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import random
from logger.logger import logger

def get_ts(datetime:str)->int:
    '''
    convert string datetime to int timestamp(milli second)
    e.g. '2021-03-05 00:00:00' -> 1614902400000
    '''
    return int(pd.Timestamp(datetime).value / 10**6)

def get_window_data(data:list or np.array, windowsize:int, stepsize:int) -> list:
    '''
    data.shape = (raw_data,)
    sample_rate = (number of samples per second)
    windowtime = window size (second)
    '''
    result = []
    for pos in range(0, len(data) - max(windowsize, stepsize) + 1, stepsize):
        result.append(data[pos:pos+windowsize])
    return result

def get_rms(data:list or np.array) -> float:
    '''
    data.shape = (raw_data,)
    '''
    result = np.power(np.power(data, 2).mean(), 1/2)
    return result

def get_unbalance(data:list or np.array) -> float:
    '''
    data.shape = (raw_data,)
    '''
    data = np.asarray(list(data))
    return ( data.max() - data.min() ) / data.mean()

def fft(data:list or np.array, sample_rate: int) -> Tuple[list, list]:
    '''
    data.shape = (raw_data,)
    '''
    data = np.asarray(list(data))
    if len(data.shape) != 1:
        raise TypeError('data must be 1-dimensional')
    _N = data.shape[-1]
    _dt = 1/sample_rate
    y_data = np.fft.fft(data)[:int(_N/2)]

    y_data = np.abs(y_data) / (_N/2)
    x_data = np.fft.fftfreq(_N, d=_dt)[:int(_N/2)]
    # logger.info("{_xf} / {_yf}".format(_xf=_xf.shape, _yf=_yf.shape))
    return list(x_data), list(y_data)

def interpolation_formatter(x_data, y_data, format:np.linspace = np.linspace(0,999, num=1000, endpoint=True), kind='linear') -> list:
    x_data = np.asarray(list(x_data))
    y_data = np.asarray(list(y_data))
    if len(x_data.shape) != 1 or len(y_data.shape) != 1:
        raise TypeError('data must be 1-dimensional')
    if x_data.shape != y_data.shape:
        raise TypeError('length of x_data and y_data should same')

    f = interp1d(x_data, y_data, kind=kind)
    return f(format)

def to_single_channel_data(data:np.array or list)->list:
    '''
    data.shape = (n_data, dim, raw_data)
    result.shape = (n_data, raw_data)
    '''
    data = np.asarray(list(data))
    result = []
    for ind in range(len(data)):
        _data = (data[ind])
        _data = _data.transpose()
        temp = []
        for _temp in _data:
            temp.append(_temp.mean())
        result.append(temp)
    return result

def single_data_slicer(data, ratio=0.8) -> Tuple[list, list]:
    '''
    ratio = ratio of train data to total data
    '''
    if ratio > 1 or ratio < 0:
        raise ValueError('ratio must be in range [0,1]')
    data = np.asarray(data)
    n_train = int(data.shape[0] * ratio)
    n_valid = int(data.shape[0] - n_train)
    bool_list = []
    bool_list.extend([True] * n_train)
    bool_list.extend([False] * n_valid)
    random.shuffle(bool_list)

    train = data[bool_list]
    
    for ind in range(len(bool_list)):
        bool_list[ind] = not bool_list[ind]

    valid = data[bool_list]

    return train.tolist(), valid.tolist()

def data_slicer(x_data:list or np.array, y_data:list or np.array=None, ratio:float=0.8, shuffle=False) -> Tuple[list,list,list,list]:
    '''
    ratio = ratio of train data to total data
    '''
    if ratio > 1 or ratio < 0:
        raise ValueError('ratio must be in range [0,1]')
    if y_data is None:
        return single_data_slicer(x_data, ratio)
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    if x_data.shape[0] != y_data.shape[0] :
        raise TypeError('x_data and y_data length must be same')

    n_train = int(x_data.shape[0] * ratio)
    n_valid = int(x_data.shape[0] - n_train)
    
    bool_list = []
    bool_list.extend([True] * n_train)
    bool_list.extend([False] * n_valid)
    if shuffle:
        random.shuffle(bool_list)
    x_train = x_data[bool_list]
    y_train = y_data[bool_list]

    for ind in range(len(bool_list)):
        bool_list[ind] = not bool_list[ind]
    x_valid = x_data[bool_list]
    y_valid = y_data[bool_list]
    return x_train.tolist(), y_train.tolist(), x_valid.tolist(), y_valid.tolist()

def to_type(data:list, _type=float)->list:
    '''
    data.shape = (phase, raw)
    '''
    result = []
    for _phase_data in data:
        result.append([])
        for entity in _phase_data:
            result[-1].append(_type(entity))
    return result
