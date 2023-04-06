from ai_core.algorithm.model import ai_model

import torch
from torch import nn
import matplotlib.pyplot as plt

from collections import OrderedDict

# class RNN(nn.Module, ai_model):
class LSTM(nn.Module):
    '''
    parameter_list
        input_size : 
        hidden_size : 
        n_layers : 
        bidirectional : 
        n_epochs : 
        dropout : 
        learning_rate : 
        device : 
    '''
    @classmethod
    def check_parameters(parameters:dict) -> bool:
        parameter_list = [
            'input_size',
            'hidden_size',
            'num_layers',
            'nonlinearity',
            'bias',
            'batch_first',
            'dropout',
            'bidirectional',
            'num_epochs',
            'learning_rate',
            'device'
            ]
        parameter_check = list( set(parameter_list) - set(parameters.keys()) )
        if parameter_check:
            return True
        else:
            return False

    def __init__(self, parameters):
        RNN.check_parameters(parameters)
        pass



class sequential_rnn():
    def __init__(self, input_dim:int, output_dim:int, hidden_dim:int, n_layers=1, learning_rate=1e-3):
        layers = OrderedDict()
        layers['0'] = nn.Linear(input_dim, hidden_dim)
        for i in range(1, n_layers):
            layers[str(i*2-1)] = nn.Linear(hidden_dim, hidden_dim)
            layers[str(i*2)] = nn.Tanh()
            # layers[str(i*2)] = nn.ELU()
            # layers[str(i*2)] = nn.ReLU()
        layers[str(n_layers*2)] = nn.Linear(hidden_dim, output_dim)

        return nn.Sequential(layers)

# class RNN(nn.Module):
#     # rnn_type:str, input_dim:int, output_dim:int, hidden_dim:int, n_layers=1, learning_rate=1e-3, bidirectional=False
#     def __init__(self, rnn_type:str, input_dim:int, output_dim:int, hidden_dim:int, n_layers:int, bidirectional:bool, device:str):
#         super(RNN, self).__init__()
#         rnn_type = str.lower(rnn_type)

#         self.hidden_size = hidden_dim
#         self.num_layers = n_layers
#         self.rnn_type = rnn_type
#         self.num_directions = 2 if bidirectional == True else 1
#         self.device = device
        
#         # rnn_type에 따른 recurrent layer 설정
#         if self.rnn_type == 'rnn':
#             self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
#         elif self.rnn_type == 'lstm':
#             self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
#         elif self.rnn_type == 'gru':
#             self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
#         else:
#             raise ValueError('incorrect rnn type, available type is rnn / lstm / gru')
        
#         # bidirectional에 따른 fc layer 구축
#         # bidirectional 여부에 따라 hidden state의 shape가 달라짐 (True: 2 * hidden_size, False: hidden_size)
#         self.fc = nn.Linear(self.num_directions * hidden_dim, output_dim)
        
#     def forward(self, x): # (batch_size x seq_len x input_size)
#         # initial hidden states 설정
#         h0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
#         # 선택한 rnn_type의 RNN으로부터 output 도출
#         if self.rnn_type in ['rnn', 'gru']:
#             out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
#         else:
#             # initial cell states 설정
#             c0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(self.device)
#             out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
#         out = self.fc(out[:, -1, :])
#         return out