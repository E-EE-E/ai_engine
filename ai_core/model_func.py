from config import attribute
from logger.logger import logger
from ai_core.model_handler import model_handler
from ai_core.file_manager import *

import glob
import numpy as np
import pickle
import time

str_config = 'config'
str_func = 'func'
str_meta = 'meta'
str_ai_id = 'ai_id'
str_label = 'label'
str_service = 'service'
str_algorithm = 'algorithm'
str_cols = 'cols'
str_time_range = 'time_range'
str_desc = 'desc'
str_data = 'data'
str_parameters = 'parameters'
str_learn = 'learn'
str_update = 'update'
str_predict = 'predict'
str_setter = 'setter'
str_getter = 'getter'
str_data_type = 'data_type'
str_model = 'model'

def config_check(config: dict) -> bool:
    try:
        result = True
        meta_check_list = []
        config_check_list = []

        _check = set([str_func, str_meta]) - set(config.keys())
        if _check:
            raise ValueError(f'config keys are missing : {_check}')
            result = False
            return False

        if config[str_func] == str_learn:
            meta_check_list = [str_label, str_service, str_algorithm, str_parameters]
            config_check_list = [str_meta, str_data, str_data_type]

        elif config[str_func] == str_update:
            meta_check_list = [str_ai_id, str_label, str_service]
            config_check_list = [str_meta]

        elif config[str_func] == str_predict:
            meta_check_list = [str_ai_id, str_label, str_service]
            config_check_list = [str_meta, str_data, str_data_type]

        _check = list(set(config_check_list) - set(config.keys()))
        if _check:
            if attribute.DEBUG:
                logger.error(f'config need more keys : {_check}')
            raise ValueError(f'config need more keys : {_check}')
            result = False

        _check = list(set(meta_check_list) - set(config[str_meta].keys()))
        if _check:
            if attribute.DEBUG:
                logger.error(f'config[{str_meta}] need more keys : {_check}')
            raise ValueError(f'config[{str_meta}] need more keys : {_check}')
            result = False

        return result

    except Exception as e:
        msg = f'config_check failed : {e}'
        if attribute.DEBUG:
            logger.error(msg)
        raise RuntimeError(msg)

def ai_process(config:dict):
    try:
        if attribute.DEBUG:
            logger.info(f'ai_process start : {config}')
        res = config_check(config)
        if res:
            try:
                if config[str_func] == str_learn:
                    res = ai_process_learn(config)
                elif config[str_func] == str_update:
                    res = ai_process_update(config)
                elif config[str_func] == str_predict:
                    res = ai_process_predict(config)
                else:
                    raise ValueError(f'config[{str_func}] value not available : {config[str_func]}')
            except Exception as e:
                raise RuntimeError(f'ai_process failed : {e}')
        elif not res:
            return False
        
        if attribute.DEBUG:
            logger.info(f'ai_process end : {config}')
        return res
    except Exception as e:
        msg = f'ai_process failed : {e}'
        if attribute.DEBUG:
            logger.error(msg)
        raise RuntimeError(msg)
    
def ai_process_learn(config:dict):
    '''
    generate model with given algorithm and parameters, set ai_id as given ai_id or auto generated ai_id
    fit model with given data
    save model
    return ai_id
    '''
    try:
        meta = config[str_meta]

        mh:model_handler
        # load existed model
        if str_ai_id in meta.keys() and meta[str_ai_id] != None:
            mh = load(meta[str_ai_id])
            if mh==False:
                logger.info(f'model ai_id does not exist')
                return False
            mh.set(data=config[str_data], meta=meta, parameters=meta[str_parameters])

        # create new model
        else:
            # create model with given ai_id
            if str_ai_id in meta.keys():
                if check_existence(str_ai_id, case=str_model)==True:
                    msg = f'given model id already exist, use different id'
                    raise RuntimeError(msg)
                mh = model_handler(id=meta[str_ai_id], algorithm=meta[str_algorithm], parameters=meta[str_parameters], meta=meta, data=config[str_data])
            # create model with generated_uuid to ai_id
            else:
                cnt = 3
                while cnt>0:
                    _id = generate_uuid()
                    if check_existence(_id, case=str_model)==False:
                        break
                    cnt-=1
                    time.sleep(0.1)
                if cnt<=0:
                    msg = f'generate unique model id failed'
                    raise RuntimeError(msg)
                mh = model_handler(id=_id, algorithm=meta[str_algorithm], parameters=meta[str_parameters], meta=meta, data=config[str_data])
        
        mh.add_data(data_set=config[str_data], data_type=config[str_data_type])
        mh.fit()
        save(mh, case=str_model)
        if attribute.DEBUG:
            logger.info(f'model id : {mh.id}')
        return mh.id

    except Exception as e:
        msg = f'ai_process_learn failed : {e}'
        logger.error(msg)
        raise RuntimeError(msg)

def ai_process_update(config:dict):
    '''
    load model with given ai_id
    set model algorithm or parameters
    add data_set to loaded model
    save model
    '''
    try:
        if check_existence(config[str_meta][str_ai_id], case=str_model):
            mh = load(config[str_meta][str_ai_id], case=str_model)
        else:
            msg = f'No model with given ai_id'
            if attribute.DEBUG:
                logger.info(msg)
            raise ValueError(msg)
        # check_list = [str_algorithm, str_parameters, str_data, str_data_type]
        
        update_dict = {
            str_data : None,
            str_data_type : None,
            str_meta : {
                str_algorithm : None,
                str_parameters : None
            }
        }
        # TODO
        if str_data not in config.keys():
            config[str_data] = None
        if str_data_type not in config.keys():
            config[str_data_type] = None

        if str_meta in config.keys():
            if str_algorithm not in config[str_meta].keys():
                config[str_meta][str_algorithm] = None
            if str_parameters not in config[str_meta].keys():
                config[str_meta][str_parameters] = None
        elif str_meta not in config.keys():
            config[str_meta] = None

        mh.set(algorithm=config[str_meta][str_algorithm], parameters=config[str_meta][str_parameters], meta=config[str_meta])
        if update_dict[str_data] is not None and update_dict[str_data_type] is not None:
            mh.add_data(data_set=update_dict[str_data], data_type=update_dict[str_data_type])
        mh.fit()
        save(mh, case=str_model)
        return mh.id
    
    except Exception as e:
        raise RuntimeError(e)

def ai_process_predict(config:dict):
    '''
    load model with given ai_id
    predict by given data with loaded model
    return predicted result
    '''
    try:
        if check_existence(config[str_meta][str_ai_id], case=str_model):
            mh = load(config[str_meta][str_ai_id], case=str_model)
        else:
            msg = f'No model with given ai_id'
            if attribute.DEBUG:
                logger.info(msg)
            raise ValueError(msg)
        res = mh.predict(data=config[str_data], data_type=config[str_data_type], batch_size=50)
        cnt = 3
        while cnt>0:
            _id = generate_uuid()
            if check_existence(_id, case=str_model)==False:
                break
            cnt-=1
            time.sleep(0.1)
        if cnt<=0:
            msg = f'generate unique model id failed'
            raise RuntimeError(msg)
        save(res, case=str_data, id=_id)
        return _id

    except Exception as e:
        msg = f'ai_process_predict failed : {e}'
        logger.error(msg)
        raise RuntimeError(msg)