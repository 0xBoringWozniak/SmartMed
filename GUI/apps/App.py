from abc import ABC, abstractmethod
from typing import Dict

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug


class App(ABC):
	'''QT app interface'''

	def __init__(self):
		pass

	@abstractmethod
	@debug
	def finish(self):
		'''close QT app'''
		raise NotImplementedError

	@debug
	def start(self):
		'''display QT app'''
		print('Start {}'.foramt(self.__name__))
