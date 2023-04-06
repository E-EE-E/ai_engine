from logger.logger import logger
from config import attribute
from ai_core.model_func import ai_process
from pydantic import BaseModel
import numpy as np
import glob

PORT = attribute.PORT
__config = 'config'
__func = 'func'
__meta = 'meta'
__ai_id = 'ai_id'
__label = 'label'
__service = 'service'
__algorithm = 'algorithm'
__cols = 'cols'
__time_range = 'time_range'
__desc = 'desc'
__data_set = 'data_set'
__parameters = 'parameters'
__learn = 'leran'
__update = 'update'
__predict = 'predict'
__setter = 'setter'
__getter = 'getter'

class Model_Setter(BaseModel):
    '''
    label name change
        change file name and meta_data to given label
    '''
    id: str
    var: str
    value: str

    def __str__(self):
        return 'id: %s, var: %s, value: %s' % (self.id, self.var, self.value)

class Model_Getter(BaseModel):
    '''
    [get ai model list]

    id : AI Engine identifier with version ( default = 'ai_engine.001' )
    config : filter to get model list
        service : service name
        id : model unique id
    '''
    id: str
    config: dict

    def __str__(self):
        return f'id: {self.id}, config: {self.config}'

class Model_AI(BaseModel):
    '''
    meta
        id : model unique id
        label : alias of id, unique in service
        service : service name
        algorithm : model algorithm
        cols : columns names for learning
        time_range : time range of data to learning
        desc : model description
    body---------------------------------------------------------------------
    {
        id : AI Engine identifier with version ( default = 'ai_engine.001' )
        config : {
            func : function type, learn/update/predict
            meta : meta data for ai_model
            data : data to use
        }
    }
    -------------------------------------------------------------------------
    func learn:
        meta:
            [ai_id] : replace exist model with given ai_id
            label : generate model with given label name, return model ai_id if given label model already exist
            service : generate ai model with service name
            algorithm : generate model with given algorithm
            parameters : algorithm setting parameters
            [cols] : columns name for learning
            [time_range] : time range of data to learning
            [desc] : model description
        data : leran ai model with given data

    func update:
        meta:
            ai_id : model id for update
            label : model unique alias in same service
            service : service name of ai model
            [parameters] : algorithm setting parameters, replace exist parameters with given parameters
                # relearn with given parameters
            [time_range] : time range of added data to update
            [desc] : overwrite given model description
        [data] : update ai model with given data set
            # relearn with exist data with given parameters if given data Null

    func predict:
        meta:
            ai_id : model id for update
            label : model unique alias in same service
            service : generate ai model with service name
        data : data for predict
    '''
    id : str
    config: dict

    def __str__(self):
        return f'id: {self.id}, config: {self.config}'

async def ai_action(body: Model_AI):
    ai_process(body.config)
    

async def setter(body: Model_Setter):
    logger.info('run api setter')
    

async def getter(body: Model_Getter):
    # service : service name
    #     algorithm : algorithm name
    #     id : model unique id
    logger.info('run api getter')
    models_path = attribute.ROOT_DIR_AI_ENGINE+'/models'+body.config['service']
    pattern = '*.pkl'
    datapath = np.unique(glob.glob(models_path + pattern, recursive=True))
