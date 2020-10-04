from typing import Dict

import pandas as pd
import pathlib

# logging decorator
import sys
sys.path.append("..")
from logs.logger import debug


class PandasPreprocessor:
	'''Class to preprocessing any datasets'''

	def __init__(self, settings: Dict):
		self.settings = settings  # settings['data']
		self.numerics_list = ['int16', 'int32',
							  'int64', 'float16', 'float32', 'float64']

	@debug
	def __read_file(self):
		ext = pathlib.Path(self.settings['path']).suffix

		if ext == '.csv':
			self.df = pd.read_csv(self.settings['path'])

			if len(self.df.columns) <= 1:
				self.df = pd.read_csv(self.settings['path'], sep=';')

		elif ext == '.xlsx':
			self.df = pd.read_excel(self.settings['path'])

		elif ext == '.tcv':
			self.df = pd.read_excel(self.settings['path'], sep='\t')

		else:
			self.df = pd.read_csv(self.settings['path'])

	@debug
	def preprocess(self):
		self.__read_file()
		self.__fillna(self.settings['preprocessing']['fillna'])
		self.__encoding(self.settings['preprocessing']['encoding'])
		self.__scale(self.settings['preprocessing']['scaling'])

	@debug
	def __fillna(self, value):
		if type(value) != 'str':
			for col in self.df.columns:
				if self.df[col].dtype in self.numerics_list:
					self.df[col] = self.df[col].fillna(value)
				else:
					self.df[col] = self.df[col].fillna(str(value))
		elif value == 'mean':
			for col in self.df.columns:
				if self.df[col].dtype in self.numerics_list:
					self.df[col] = self.df[col].fillna(self.df[col].mean())
				else:
					self.df[col] = self.df[col].fillna(
						self.df[col].mode().values[0])
		elif value == 'median':
			for col in self.df.columns:
				if self.df[col].dtype in self.numerics_list:
					self.df[col] = self.df[col].fillna(self.df[col].median())
				else:
					self.df[col] = self.df[col].fillna(
						self.df[col].mode().values[0])
		elif value == 'droprows':
			self.df = self.df[col].dropna()

	@debug
	def __encoding(self, method):
		return self.df

	@debug
	def __scale(self, method):
		return self.df

	def get_numeric_df(self, df):
		return df.select_dtypes(include=self.numerics_list)
