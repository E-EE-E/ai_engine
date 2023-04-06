'''
manage model files
'''
from config import attribute
from logger.logger import logger
from ai_core.model_handler import model_handler

import os
import uuid
import glob
import numpy as np
import pickle

MODEL_DIRECTORY = './models'
DATA_DIRECTORY = './data'
if not os.path.isdir(MODEL_DIRECTORY):
    os.mkdir(MODEL_DIRECTORY)
if not os.path.isdir(DATA_DIRECTORY):
    os.mkdir(DATA_DIRECTORY)


def generate_uuid() -> str:
    '''
    generate uuid using host ID and current time
    return uuid as string
    '''
    try:
        return str(uuid.uuid1())
    except Exception as e:
        logger.error(f'generate uuid failed : {e}')
        return False

def check_existence(id:str, case:str) -> bool:
    '''
    check existence of given ai_id model
    return
        True: file exist with given id
        False: file not exist with given id
    '''
    try:
        if case=='model':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/models'
        elif case=='data':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/data'
        else:
            msg = f'wrong case : {case}'
            if attribute.DEBUG:
                logger.warn(msg)
            raise msg
        file_pattern = f'/{id}.pkl'
        filepath = list(np.unique(glob.glob(_path+file_pattern)))

        if len(filepath)==1:
            return True
        elif len(filepath)==0:
            return False
        elif len(filepath)>1:
            msg = f'{case} with given id exist more then one, remove id duplicates'
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            msg = f'file count error : {case} | {id} -> {len(filepath)}'
            logger.error(msg)
            raise ValueError(msg)
    
    except Exception as e:
        msg = f'check {case} existence failed : {e}'
        logger.error(msg)
        raise RuntimeError(msg)

def get_list(case:str) -> list:
    '''
    get all exist files of case id
    return id list
    '''
    try:
        if case=='model':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/models'
        elif case=='data':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/data'
        file_pattern = '/*.pkl'
        filepath = list(np.unique(glob.glob(_path+file_pattern)))

        result = []
        for _filepath in filepath:
            result.append((_filepath.split('/')[-1]).split('.')[0])
        return result

    except Exception as e:
        msg = f'get {case} list failed : {e}'
        logger.error(msg)
        raise RuntimeError(msg)

def load(id:str, case:str) -> model_handler:
    '''
    load file with given id
    return loaded object
    '''
    try:
        if check_existence(id=id, case=case)==False:
            logger.info(f'file does not exist: {case} | {id}')
            return False
        
        if case=='model':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/models'
        elif case=='data':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/data'
        file_pattern = f'/{id}.pkl'
        filepath = _path+file_pattern
        f = open(filepath, 'r+b')
        _model = pickle.load(f)
        return _model

    except Exception as e:
        msg = f'load {case} failed : {e}'
        logger.error(msg)
        raise RuntimeError(msg)

def remove(ai_id:str, case:str) -> bool:
    '''
    remove file with given id and case
    return
        True: success to remove file
        False: fail to remove file
    '''
    try:
        if check_existence(id=ai_id)==False:
            logger.info(f'model ai_id does not exist')
            return False
        
        if case=='model':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/models'
        elif case=='data':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/data'
        file_pattern = f'/{ai_id}.pkl'
        filepath = _path+file_pattern
        os.system(f'rm -f {filepath}')
        return True

    except Exception as e:
        msg = f'remove Bai model failed : {e}'
        logger.error(msg)
        raise RuntimeError(msg)
    

def copy(from_id:str, to_id:str, case:str) -> bool:
    '''
    copy exist file to given id
    return
        True: success to copy file
        False: fail to copy file
    '''
    try:
        if check_existence(id=from_id)==False:
            logger.info(f'{case} from_id does not exist')
            return False
        if check_existence(id=to_id)==True:
            logger.info(f'{case}l to_id already exists.')
            return False
        
        if case=='model':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/models'
        elif case=='data':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/data'
        from_file_pattern = f'/{from_id}.pkl'
        to_file_pattern = f'/{to_id}.pkl'
        os.system(f'cp {_path + from_file_pattern} {_path + to_file_pattern}')
        return True

    except Exception as e:
        msg = f'copy file failed : {e}'
        logger.error(msg)
        raise RuntimeError(msg)
    

def rename(from_id:str, to_id:str, case:str):
    '''
    rename exist ai model to given ai_id
    return
        True: success to rename ai model
        False: fail to rename ai model
    '''
    try:
        if check_existence(id=from_id)==False:
            logger.info(f'model from_id does not exist')
            return False
        if check_existence(id=to_id)==True:
            logger.info(f'model to_id already exists.')
            return False
        
        if case=='model':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/models'
        elif case=='data':
            _path = attribute.ROOT_DIR_AI_ENGINE+'/data'
        from_file_pattern = f'/{from_id}.pkl'
        to_file_pattern = f'/{to_id}.pkl'
        os.system(f'mv {_path + from_file_pattern} {_path + to_file_pattern}')
        return True
        
    except Exception as e:
        msg = f'rename ai model failed : {e}'
        logger.error(msg)
        raise RuntimeError(msg)
    
def save(obj, case:str, id:str=None) -> bool:
    '''
    return True if save ai model success else False
    '''
    try:
        if case=='model':
            _path = f'{attribute.ROOT_DIR_AI_ENGINE}/models/{obj.id}.pkl'
        elif case=='data':
            if id==None:
                msg = f'id is required to save data'
                raise ValueError(msg)
            _path = f'{attribute.ROOT_DIR_AI_ENGINE}/data/{id}.pkl'
        with open(_path, 'w+b') as f:
            pickle.dump(obj, f)
        return _path
    except Exception as e:
        raise RuntimeError(f'save file failed : {e}')