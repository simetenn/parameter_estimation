from __future__ import absolute_import, division, print_function, unicode_literals

"""
Small utility functions for various purposes.
"""

__all__ = ["lengths", "none_to_nan", "contains_nan", "is_regular",
            "MyFormatter", "TqdmLoggingHandler", "MultiprocessLoggingHandler",
            "setup_module_logger", "setup_logger",
           "has_handlers", "add_file_handler", "add_screen_handler"]

from .logger import setup_module_logger, setup_logger
from .logger import has_handlers, add_file_handler, add_screen_handler
from .logger import MyFormatter, TqdmLoggingHandler, MultiprocessLoggingHandler
from .utility import lengths, none_to_nan, contains_nan
from .utility import is_regular, set_nan
from .utility import create_model_parameters
