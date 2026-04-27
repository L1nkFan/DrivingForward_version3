from .logger import Logger
from .misc import get_config

__all__ = ['Logger', 'get_config']


import sys


_LIBS = ['./external/packnet_sfm', './external/dgp', './external/monodepth2']    

def setup_env():       
    if not _LIBS[0] in sys.path:        
        for lib in _LIBS:
            sys.path.append(lib)

setup_env()