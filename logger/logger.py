import logging
import logging.handlers
import os
from config import attribute

LOG_DIRECTORY = './log'
LOG_FILE_BACKUP_COUNT = attribute.LOG_FILE_BACKUP_COUNT

if not os.path.isdir(LOG_DIRECTORY):
        os.mkdir(LOG_DIRECTORY)

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(funcName)s - %(message)s')

file_handler = logging.handlers.TimedRotatingFileHandler(filename=LOG_DIRECTORY + '/log.log', when='midnight', interval=1,  encoding='utf-8', backupCount=LOG_FILE_BACKUP_COUNT)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()