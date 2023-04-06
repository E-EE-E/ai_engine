from config import attribute
from logger.logger import logger
from api.api_server import serve

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    serve()