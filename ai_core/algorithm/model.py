from logger.logger import logger, printProgressBar
from config import attribute
from preprocessor.preprocessor import data_slicer
from ai_core.algorithm.RNN import RNN, rnn, gru, lstm
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
import pickle
import time
import copy

from ai_core.algorithm.model import *
str_input_size = 'input_size'
str_output_size = 'output_size'
str_hidden_size = 'hidden_size'
str_num_layers = 'num_layers'
str_nonlinearity = 'nonlinearity'
str_bias = 'bias'
str_batch_first = 'batch_first'
str_dropout = 'dropout'
str_bidirectional = 'bidirectional'
str_num_epochs = 'num_epochs'
str_learning_rate = 'learning_rate'
str_device = 'device'
str_num_directions = 'num_directions'
# str_data_path = 'data_path'
str_data_type = 'data_type'
str_batch_size = 'batch_size'
model_list = {
    'rnn' : rnn,
    'lstm' : lstm,
    'gru' : gru
}
class ai_model:
    '''
    core model handler
    handling algorithm list : [rnn, gru, lstm]
    parameter_list
        input_size: int
            number of features of input, 0 < input_size
        output_size: int
            number of features of output, 0 < output_size
        hidden_size: int
            hidden layer size, 0 < hidden_size
        n_layers: int
            number of layers, 0 < n_layers
        bidirectional: bool
            bidirectional, {True | False}
        n_epochs: int
            number of learning iteration, 0 < n_epochs
        dropout: float
            ratio of dropout, 0 < dropout < 1
        learning_rate:float
            learning rate, 0 < learning_rate < 0.1
        device: str
            device for learning, {'cpu' | 'cuda'}

        data_set.shape = (n, 2)
        [
            [    [target]        [data]    ],
            [    [target]        [data]    ],
            [    [target]        [data]    ],
            ... ,
        ]
    '''
    def __init__(self, algorithm:str, config:dict):
        if not self.check_config(config):
            msg = f'check config failed'
            raise ValueError(msg)
        self.config = config

        self.model:RNN = model_list[algorithm](config).to(config[str_device])
        self.dataset: list = None
        self.train_dataloader: torch.utils.data.DataLoader
        self.valid_dataloader: torch.utils.data.DataLoader
    
    def check_config(self, config:dict) -> bool:
        key_list = [
            str_input_size, 
            str_output_size, 
            str_hidden_size, 
            str_num_layers, 
            str_bidirectional, 
            str_num_epochs, 
            str_dropout, 
            str_learning_rate, 
            str_device, 
            str_batch_size
        ]
        if set(key_list) - set(config.keys()):
            return False

        config[str_device] = 'cuda' if ( torch.cuda.is_available() & (config[str_device] == 'cuda') ) else 'cpu'
        return True

    def check_data_shape(self, data1:list, data2:list):
        '''
        data_set.shape = (n, 2)
        [
            [    [target]        [data]    ],
            [    [target]        [data]    ],
            [    [target]        [data]    ],
            ... ,
        ]
        '''
        if data1 is None or data2 is None:
            return True
        target_check = np.asarray(data1[0][0]).shape == np.asarray(data2[0][0]).shape
        data_check = np.asarray(data1[0][1]).shape == np.asarray(data2[0][1]).shape
        return target_check and data_check

    def add_data(self, data:str or pd.DataFrame, data_type):
        '''
        data : datapath to add to self.dataset
        add data to exist self.data_set with given data path or data frame
        data shape must be same to exist data_set
        '''
        try:
            df: list
            df = self.load_data(data=data, data_type=data_type)
            if attribute.DEBUG:
                logger.info(f'add data shape : {np.asarray(df).shape}')

            if self.dataset==None:
                self.dataset = df
                return True
            else:
                if self.check_data_shape(self.dataset, df):
                    self.dataset += df
                    return True
                else:
                    if attribute.DEBUG:
                        logger.warn(f'added data shape is not same as existing data')
                    return False
        except Exception as e:
            msg = f'{e}'
            logger.error(msg)
            raise RuntimeError(msg)
        
    def remove_data(self):
        self.dataset = None
    
    def load_data(self, data:str or list or pd.DataFrame, data_type)->list:
        '''
        data_type = { 'csv' | 'pkl' | 'df' | 'list'}
        '''
        try:
            if attribute.DEBUG:
                logger.info(f'data : [{data_type}] {data}')
            df: list
            if data_type == 'csv':
                df = pd.read_csv(data).values.tolist()
            elif data_type == 'pkl':
                f = open(data, 'r+b')
                df = pickle.load(f)
                if type(df) is list:
                    pass
                elif type(df) is pd.DataFrame:
                    df = df.values.tolist()
                else:
                    msg = f'wrong data type in file, data type should be list : {type(df)}'
                    raise ValueError(msg)
            elif data_type == 'df':
                df = pd.DataFrame(data).values.tolist()
            elif data_type == 'list':
                pass
            else:
                msg = f'wrong data_type, data_type should be list : {data_type}'
                raise ValueError(msg)
                
            return df
        except Exception as e:
            raise RuntimeError(e)

    def get_loader(self, data, batch_size = 50):
        try:
            train_data, valid_data = data_slicer(data, ratio=0.8)
            if attribute.DEBUG:
                logger.info(f'get_loader train shape : {np.asarray(train_data).shape} | {np.asarray(train_data[0][0]).shape} | {np.asarray(train_data[0][1]).shape}')
                logger.info(f'get_loader valid shape : {np.asarray(valid_data).shape} | {np.asarray(valid_data[0][0]).shape} | {np.asarray(valid_data[0][1]).shape}')

            train_dataset = TrainDataset(train_data)
            valid_dataset = TrainDataset(valid_data)
            self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            return self.train_dataloader, self.valid_dataloader
        except Exception as e:
            msg = f'model.get_loader failed : {e}'
            raise RuntimeError(msg)

    def fit(self):
        try:
            self.get_loader(self.dataset, batch_size=self.config[str_batch_size])
            self.train(train_dataloader=self.train_dataloader, valid_dataloader=self.valid_dataloader)
        except Exception as e:
            msg = f'model.fit failed : {e}'
            raise RuntimeError(msg)

    def train(self, train_dataloader, valid_dataloader):
        try:
            _time = time.time()
            best_model_wts = copy.deepcopy(self.model.state_dict())
            best_val_loss = False
            optimizer = optim.Adam(self.model.parameters(), lr=self.config[str_learning_rate])
            criterion = nn.MSELoss()

            for epoch in range(self.config[str_num_epochs]):
                printProgressBar (epoch+1, self.config[str_num_epochs], prefix = '[Train]', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r")

                # 각 epoch마다 순서대로 training과 validation을 진행
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()  # 모델을 training mode로 설정
                        dataloader = train_dataloader
                    else:
                        self.model.eval()   # 모델을 validation mode로 설정
                        dataloader = valid_dataloader
                    running_loss = 0.0
                    running_total = 0

                    # training과 validation 단계에 맞는 dataloader에 대하여 학습/검증 진행
                    for inputs, labels in dataloader:
                        inputs = inputs.to(self.config[str_device])
                        labels = labels.to(self.config[str_device])

                        # parameter gradients를 0으로 설정
                        optimizer.zero_grad()

                        # forward
                        # training 단계에서만 gradient 업데이트 수행
                        with torch.set_grad_enabled(phase == 'train'):
                            # input을 model에 넣어 output을 도출한 후, loss를 계산함
                            outputs = self.model(inputs)
                            # logger.info(f'\ninputs.shape : {inputs.shape}\noutputs.shape : {outputs.shape}\nlabels.shape : {labels.shape}')
                            loss = criterion(outputs.unsqueeze(2), labels[:, :, -1:])

                            # backward (optimize): training 단계에서만 수행
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # batch별 loss를 축적함
                        running_loss += loss.item() * inputs.size(0)
                        running_total += labels.size(0)

                    # epoch의 loss 및 RMSE 도출
                    epoch_loss = running_loss / running_total
                    epoch_rmse = np.sqrt(epoch_loss)

                    # if epoch == 0 or (epoch + 1) % 10 == 0:
                    #     print('{} Loss: {:.4f} MSE: {:.4f}'.format(phase, epoch_loss, epoch_rmse))

                    # validation 단계에서 validation loss가 감소할 때마다 best model 가중치를 업데이트함
                    if best_val_loss is False or ( phase == 'val' and epoch_loss < best_val_loss ):
                        best_val_loss = epoch_rmse
                        best_model_wts = copy.deepcopy(self.model.state_dict())

            # 전체 학습 시간 계산
            time_elapsed = time.time() - _time
            print(f'\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val MSE: {best_val_loss:4f}')

            # validation loss가 가장 낮았을 때의 best model 가중치를 불러와 best model을 구축함
            self.model.load_state_dict(best_model_wts)
            return self.model
        
        except Exception as e:
            msg = f'model.train failed : {e}'
            raise RuntimeError(msg)

    def predict(self, data:str or pd.DataFrame or list, data_type:str, batch_size=500)-> list:
        '''
        data_type = { 'csv' | 'pkl' | 'df' | 'list'}
        '''
        try:
            df:list = self.load_data(data=data, data_type=data_type)
            if attribute.DEBUG:
                logger.info(f'df shape : {np.asarray(df).shape}')
            pred_dataset = PredictDataset(df)
            pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
            self.model.eval()

            preds = []
            with torch.no_grad():
                for inputs in pred_loader:
                    inputs = inputs.to(self.config[str_device])
                    outputs = self.model(inputs)
                    preds.append(outputs.detach().cpu().numpy())
            preds = np.concatenate(preds)
            preds = np.expand_dims(preds, axis=-1)
            preds = preds.tolist()
            return preds
        except Exception as e:
            raise RuntimeError(f'predict failed : {e}')

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data:list=None):
        '''
        data_set.shape = (n, 2)
        [
            [    [target]      [data]    ],
            [    [target]      [data]    ],
            [    [target]      [data]    ],
            ... ,
        ]
        '''
        try:
            self.x = np.asarray(data)[:, 1].tolist()
            self.y = np.asarray(data)[:, 0].tolist()
        except Exception as e:
            msg = f'initalize dataset failed : {e}'
            raise ValueError(msg)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.FloatTensor(self.y[idx])
        return x, y
    
class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, data:list=None):
        '''
        data_set.shape = (n, data.shape)
        [
            [ data ],
            [ data ],
            [ data ],
            ... ,
        ]
        '''
        try:
            self.x = np.asarray(data)[:,0].tolist()
        except Exception as e:
            msg = f'initalize dataset failed : {e}'
            raise ValueError(msg)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        x = torch.FloatTensor(self.x[idx])
        return x