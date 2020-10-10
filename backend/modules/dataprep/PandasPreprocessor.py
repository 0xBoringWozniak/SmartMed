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
		self.settings = settings # settings['data']

	@debug
	def preprocess(self):
		if self.settings['preprocessing']['AUTO']:
			return pd.read_csv(self.settings['path'], sep=',')
		else:
			self.df = pd.read_csv(self.settings['path'], sep=';')
			self.__fillna(self.settings['preprocessing']['fillna'])
			self.__encoding(self.settings['preprocessing']['encoding'])
			self.__scale(self.settings['preprocessing']['scaling'])

	@debug
	def __fillna(self, value):
		if type(value) != 'str':
				for col in self.df.columns:
					if self.df[col].dtype in {'float64', 'float32', 'int64', 'int32'}:
						self.df[col] = self.df[col].fillna(value)
					else:
						self.df[col] = self.df[col].fillna(str(value))
		elif value == 'mean':
			for col in self.df.columns:
				if self.df[col].dtype in {'float64', 'float32', 'int64', 'int32'}:
					self.df[col] = self.df[col].fillna(self.df[col].mean())
				else:
					self.df[col] = self.df[col].fillna(self.df[col].mode().values[0])
		elif value == 'median':
			for col in self.df.columns:
				if self.df[col].dtype in {'float64', 'float32', 'int64', 'int32'}:
					self.df[col] = self.df[col].fillna(self.df[col].median())
				else:
					self.df[col] = self.df[col].fillna(self.df[col].mode().values[0])
		elif value == 'droprows':
			self.df = self.df[col].dropna()

	@debug
	def __encoding(self, method):
		return self.df

	@debug
	def __scale(self, method):
		return self.df
