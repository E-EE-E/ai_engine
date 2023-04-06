import torch
from torch import nn
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

class RNN(nn.Module):
    '''
    parameter_list
        input_size : number of features of input
        output_size : number of features of output
        hidden_size : 
        n_layers : 
        bidirectional : 
        n_epochs : 
        dropout : 
        learning_rate : 
        device : 
    '''

    def __init__(self, config:dict):
        super().__init__()
        self.config = config
        self.config[str_num_directions] = 2 if config[str_bidirectional]==True else 1
        self.fc = nn.Linear(self.config[str_num_directions] * config[str_hidden_size], config[str_output_size])

    def check_config(self, config:dict):
        pass

    def forward(self, x): # (batch_size x seq_len x input_size)
        # initial hidden states 설정
        pass

class rnn(RNN):
    def __init__(self, config):
        super().__init__(config=config)
        self.model = nn.RNN(
            input_size = self.config[str_input_size], 
            hidden_size = self.config[str_hidden_size], 
            num_layers = self.config[str_num_layers], 
            batch_first = self.config[str_batch_first], 
            bidirectional=self.config[str_bidirectional]
        )
    def forward(self, x):
        h0 = torch.zeros(self.config[str_num_directions] * self.config[str_num_layers], x.size(0), self.config[str_hidden_size]).to(self.config[str_device])
        out, _ = self.model(x, h0)
        out = self.fc(out[:,-1,:])
        return out
    
class gru(RNN):
    def __init__(self, config):
        print(config)
        super(gru, self).__init__(config=config)
        self.model = nn.GRU(
            input_size = self.config[str_input_size], 
            hidden_size = self.config[str_hidden_size], 
            num_layers = self.config[str_num_layers], 
            batch_first = self.config[str_batch_first], 
            bidirectional=self.config[str_bidirectional]
        )
    def forward(self, x):
        h0 = torch.zeros(self.config[str_num_directions] * self.config[str_num_layers], x.size(0), self.config[str_hidden_size]).to(self.config[str_device])
        out, _ = self.model(x, h0)
        out = self.fc(out[:,-1,:])
        return out
    
class lstm(RNN):
    def __init__(self, config):
        super().__init__(config=config)
        self.model = nn.LSTM(
            input_size = self.config[str_input_size], 
            hidden_size = self.config[str_hidden_size], 
            num_layers = self.config[str_num_layers], 
            batch_first = self.config[str_batch_first], 
            bidirectional=self.config[str_bidirectional]
        )
    def forward(self, x):
        h0 = torch.zeros(self.config[str_num_directions] * self.config[str_num_layers], x.size(0), self.config[str_hidden_size]).to(self.config[str_device])
        c0 = torch.zeros(self.config[str_num_directions] * self.config[str_num_layers], x.size(0), self.config[str_hidden_size]).to(self.config[str_device])
        out, _ = self.model(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out