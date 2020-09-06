from typing import Dict

import pandas as pd

import logging

logging.basicConfig(filename='~/../logs/start.log', level=logging.DEBUG)


def debug(fn):
	'''logging decorator'''
	def wrapper(*args, **kwargs):
		logging.debug("Entering {:s}...".format(fn.__name__))
		result = fn(*args, **kwargs)
		logging.debug("Finished {:s}.".format(fn.__name__))
		return result

	return wrapper


class PandasPreprocessor:
	'''Class to preprocessing any datasets'''
	def __init__(self, settings: Dict):
		self.settings = settings
