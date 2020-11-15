from abc import ABC, abstractmethod
from typing import Dict

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug

from .dataprep import PandasPreprocessor


class Module(ABC):

	def __init__(self, settings: Dict):
		self.settings = settings

		# preprocessor to manipulate with data
		self.pp = PandasPreprocessor(settings['data'])
		super().__init__()

	@debug
	def _prepare_data(self):
		'''manipulate data according to settings'''
		raise NotImplementedError

	@abstractmethod
	def _prepare_dashboard_settings(self):
		'''
		construct dashboard settings
		here you can start model or make any other calculations
		'''
		raise NotImplementedError

	@abstractmethod
	def _prepare_dashboard(self):
		'''generate dash using DashContsructor'''
		raise NotImplementedError

	@debug
	def run(self):
		'''
		start module
		1. take settings
		2. manipulate data (prepare data, build models and calculate metrics)
		3. prepare dash
		4. start dash
		'''
		self.data = self._prepare_data()
		self.settings = self._prepare_dashboard_settings()
		self._prepare_dashboard()
		self.start()
