import sys
import logging

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Creating the lime logger
_logger = logging.getLogger("pySB99")
_logger.setLevel(logging.INFO)

# Outputting format
consoleHandle = logging.StreamHandler()
consoleHandle.setFormatter(logging.Formatter('%(name)s %(levelname)s: %(message)s'))
_logger.addHandler(consoleHandle)

# Get python version being used
__python_version__ = sys.version_info

from pystarburst99.ins_outs import pySB99_cfg
from pystarburst99.workflow import Compiler

# Library version
__version__ = pySB99_cfg['metadata']['version']

# Logging configuration
_logger.debug(f'Launching pyStarburst99 {__version__} in Python {__python_version__}')



