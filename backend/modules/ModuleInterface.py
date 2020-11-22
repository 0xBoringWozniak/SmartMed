from abc import ABC, abstractmethod
from typing import Dict

# logging decorator
import sys
sys.path.append("...")
from logs.logger import debug

from .dataprep import PandasPreprocessor


class Module(ABC):

	def __init__(self, settings: Dict):

		# get settings['MODULE_SETTINGS']
		self.settings = settings

		# custom class preprocessor with pandas
		self.pp = PandasPreprocessor(settings['data'])

		# call dash.Dashboard __init__ to crate Dashboard object
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
		A standard instructuion to start module

		1. prepare data using dataprep.PandasPreprocessor
		2. prepare maths models from models, manipulating with settings to dashboard
		3. prepare dashboard (if it is needed)
		4. start dashboard on localhost

		'''

		# create .data attribute in Module
		self.data = self._prepare_data()

		# inner settings and backend calculations
		self.settings = self._prepare_dashboard_settings()

		# dashboard Dash manipulating
		self._prepare_dashboard()

		# dashboard start implements from dash.Dashboard
		# use dash.Dashboard instruction
		self.start()
